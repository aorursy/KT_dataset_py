!pip install pytorch-pfn-extras==0.3.0
import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
import zarr



import l5kit

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset



from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

from l5kit.evaluation import write_pred_csv



from matplotlib import animation, rc

from IPython.display import HTML



rc('animation', html='jshtml')

print("l5kit version:", l5kit.__version__)
import torch

from pathlib import Path



import pytorch_pfn_extras as ppe

from math import ceil

from pytorch_pfn_extras.training import IgniteExtensionsManager

from pytorch_pfn_extras.training.triggers import MinValueTrigger



from torch import nn, optim

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Subset

import pytorch_pfn_extras.training.extensions as E
# --- Model utils ---

import torch

from torchvision.models import resnet18

from torch import nn

from typing import Dict





class LyftMultiModel(nn.Module):



    def __init__(self, cfg: Dict, num_modes=3):

        super().__init__()



        # TODO: support other than resnet18?

        backbone = resnet18(pretrained=True, progress=True)

        self.backbone = backbone



        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels



        self.backbone.conv1 = nn.Conv2d(

            num_in_channels,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False,

        )



        # This is 512 for resnet18 and resnet34;

        # And it is 2048 for the other resnets

        backbone_out_features = 512



        # X, Y coords for the future positions (output shape: Bx50x2)

        self.future_len = cfg["model_params"]["future_num_frames"]

        num_targets = 2 * self.future_len



        # You can add more layers here.

        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=backbone_out_features, out_features=4096),

        )



        self.num_preds = num_targets * num_modes

        self.num_modes = num_modes



        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)



    def forward(self, x):

        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)



        x = self.head(x)

        x = self.logit(x)



        # pred (bs)x(modes)x(time)x(2D coords)

        # confidences (bs)x(modes)

        bs, _ = x.shape

        pred, confidences = torch.split(x, self.num_preds, dim=1)

        pred = pred.view(bs, self.num_modes, self.future_len, 2)

        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences



    
# --- Utils ---

import yaml





def save_yaml(filepath, content, width=120):

    with open(filepath, 'w') as f:

        yaml.dump(content, f, width=width)





def load_yaml(filepath):

    with open(filepath, 'r') as f:

        content = yaml.safe_load(f)

    return content





class DotDict(dict):

    """dot.notation access to dictionary attributes



    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767

    """  # NOQA



    __getattr__ = dict.get

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__



    
# Referred https://www.kaggle.com/pestipeti/pytorch-baseline-inference

def run_prediction(predictor, data_loader):

    predictor.eval()



    pred_coords_list = []

    confidences_list = []

    timestamps_list = []

    track_id_list = []



    with torch.no_grad():

        dataiter = tqdm(data_loader)

        for data in dataiter:

            image = data["image"].to(device)

            # target_availabilities = data["target_availabilities"].to(device)

            # targets = data["target_positions"].to(device)

            pred, confidences = predictor(image)



            pred_coords_list.append(pred.cpu().numpy().copy())

            confidences_list.append(confidences.cpu().numpy().copy())

            timestamps_list.append(data["timestamp"].numpy().copy())

            track_id_list.append(data["track_id"].numpy().copy())

    timestamps = np.concatenate(timestamps_list)

    track_ids = np.concatenate(track_id_list)

    coords = np.concatenate(pred_coords_list)

    confs = np.concatenate(confidences_list)

    return timestamps, track_ids, coords, confs

# --- Lyft configs ---

cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },



    'raster_params': {

        'raster_size': [224, 224],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },



    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 4

    },



    'valid_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 32,

        'shuffle': False,

        'num_workers': 4

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },



    'train_params': {

        'max_num_steps': 10000,

        'checkpoint_every_n_steps': 5000,



        # 'eval_every_n_steps': -1

    }

}

flags_dict = {

    "debug": False,

    # --- Data configs ---

    "l5kit_data_folder": "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",

    # --- Model configs ---

    "pred_mode": "multi",

    # --- Training configs ---

    "device": "cuda:0",

    "out_dir": "results/multi_train",

    "epoch": 2,

    "snapshot_freq": 50,

}
flags = DotDict(flags_dict)

out_dir = Path(flags.out_dir)

os.makedirs(str(out_dir), exist_ok=True)

print(f"flags: {flags_dict}")

save_yaml(out_dir / 'flags.yaml', flags_dict)

save_yaml(out_dir / 'cfg.yaml', cfg)

debug = flags.debug
# set env variable for data

l5kit_data_folder = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder

dm = LocalDataManager(None)



print("Load dataset...")

default_test_cfg = {

    'key': 'scenes/test.zarr',

    'batch_size': 32,

    'shuffle': False,

    'num_workers': 4

}

test_cfg = cfg.get("test_data_loader", default_test_cfg)



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



test_path = test_cfg["key"]

print(f"Loading from {test_path}")

test_zarr = ChunkedDataset(dm.require(test_path)).open()

print("test_zarr", type(test_zarr))

test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]

test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataset = test_agent_dataset

if debug:

    # Only use 100 dataset for fast check...

    test_dataset = Subset(test_dataset, np.arange(100))

test_loader = DataLoader(

    test_dataset,

    shuffle=test_cfg["shuffle"],

    batch_size=test_cfg["batch_size"],

    num_workers=test_cfg["num_workers"],

    pin_memory=True,

)



print(test_agent_dataset)

print("# AgentDataset test:", len(test_agent_dataset))

print("# ActualDataset test:", len(test_dataset))
device = torch.device(flags.device)



if flags.pred_mode == "multi":

    predictor = LyftMultiModel(cfg)

else:

    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")



pt_path = "/kaggle/input/lyft-resnet18-baseline/predictor.pt"

print(f"Loading from {pt_path}")

predictor.load_state_dict(torch.load(pt_path))

predictor.to(device)
# --- Inference ---

timestamps, track_ids, coords, confs = run_prediction(predictor, test_loader)
# Draw lottery to adjust score

# coords[-5:] += np.random.uniform(low=0.0, high=1.0, size=coords[-5:].shape)
csv_path = "submission.csv"

write_pred_csv(

    csv_path,

    timestamps=timestamps,

    track_ids=track_ids,

    coords=coords,

    confs=confs)

print(f"Saved to {csv_path}")