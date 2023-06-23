import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

print(sys.path, '<< sys path')

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split],
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        args = args,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC,
        proposal_generator=args.proposal_generator,
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained VoteNet...")
        pretrained_model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=True,
            dataset_config=DC
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False

            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False

    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    # different lr for various modules.
    if args.proposal_generator == "votenet":
        weight_dict = {
            'detr': {'lr': 0.0001},
            'lang': {'lr': 0.0005},
            'match': {'lr': 0.0005},
        }
    else:
        # import ipdb; ipdb.set_trace()
        model_3detr_keys = torch.load("pretrained_3detr/scannet_masked_ep1080.pth")['model'].keys()
        weight_dict = {"model." + k : {'lr' : 0.00001} for k in model_3detr_keys}
        weight_dict['model'] = {'lr': 0.001}
        weight_dict['lang'] = {'lr': 0.0005}
        weight_dict['match'] = {'lr': 0.0005}
    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    # params = model.parameters()
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    if args.coslr:
        LR_DECAY_STEP = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-5,
        }
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE, BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
    solver = Solver(
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        args=args,
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, lang_num_max):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)
                """
                if data["scene_id"] not in scanrefer_train_new:
                    scanrefer_train_new[data["scene_id"]] = []
                scanrefer_train_new[data["scene_id"]].append(data)
                """
        scanrefer_train_new.append(scanrefer_train_new_scene)

        new_scanrefer_val = scanrefer_val
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer_val:
            # if data["scene_id"] not in scanrefer_val_new:
            # scanrefer_val_new[data["scene_id"]] = []
            # scanrefer_val_new[data["scene_id"]].append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        scanrefer_val_new.append(scanrefer_val_new_scene)

    print("scanrefer_train_new", len(scanrefer_train_new), len(scanrefer_val_new), len(scanrefer_train_new[0]))  # 4819 1253 8
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    print("training sample numbers", sum)  # 36665
    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))  # 36665 9508

    return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    scanrefer_new = {
        "train": scanrefer_train_new,
        "val": scanrefer_val_new
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "train", DC, augment=True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "val", DC, augment=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    ### 3DETR Arguments
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    parser.add_argument("--proposal_generator", default="votenet", type=str)
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=3,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

