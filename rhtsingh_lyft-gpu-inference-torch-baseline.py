!pip install --no-index -f ../input/kaggle-l5kit pip==20.2.2 >/dev/nul

!pip install --no-index -f ../input/kaggle-l5kit -U l5kit > /dev/nul
import os

import gc

gc.enable()



import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torch.utils.data import DataLoader



from tqdm import tqdm 

from typing import Dict

from pathlib import Path

from prettytable import PrettyTable



from l5kit.configs import load_config_data

from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer

from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset

from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS

from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

from l5kit.geometry import transform_points

from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
DIR_INPUT = '../input/lyft-motion-prediction-autonomous-vehicles/'

DIR_INPUT_TRAIN = '../input/lyft-full-training-set/'

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_submission.csv"



DEBUG = False



os.environ['L5KIT_DATA_FOLDER'] = DIR_INPUT

dm = LocalDataManager(None)



cfg = {

    'format_version':4,

    'model_params':{

        'model_architecture':'resnet18',

        'history_num_frames':15,

        'history_step_size':1,

        'history_delta_time':0.1,

        'future_num_frames':50,

        'future_step_size':1,

        'future_delta_time':0.1

    },

    'raster_params':{

        'raster_size':[331,331],

        'pixel_size':[0.5,0.5],

        'ego_center':[0.25,0.25],

        'map_type':'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    'train_data_loader':{

        'key':'scenes/train.zarr',

        'batch_size':16,

        'shuffle':True,

        'num_workers':4

    },

    'train_params':{

        'max_num_steps': 1000 if DEBUG else 20000,

        'checkpoint_every_n_steps':5000

    },

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    }

}
class ResNetModel(nn.Module):

    def __init__(self, cfg):

        super(ResNetModel, self).__init__()

        # load pre-trained Conv2D model

        self.backbone = torchvision.models.resnet18(pretrained=False, progress=False)

        self.backbone.load_state_dict(

            torch.load(

                '../input/resnet18/resnet18.pth'

            )

        )

        # change input channels number to match the rasterizer's output

        num_history_channels = (cfg['model_params']['history_num_frames']+1) * 2

        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(

            num_in_channels,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False

        )

        

        # change output size to (X, Y) * number of future states

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        self.backbone.fc = nn.Linear(in_features=512, out_features=num_targets)        

    

    def forward(self, x):

        # Forward pass

        return self.backbone(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetModel(cfg).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")

print("Device is {}.".format(device))
WEIGHT_FILE = '../input/gpubaseline/model_state_14999.pth'



test_cfg = cfg["test_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])





print(test_dataloader)
model = ResNetModel(cfg=cfg).to(device)

if WEIGHT_FILE is not None:

    model.load_state_dict(

        torch.load(WEIGHT_FILE, map_location=device),

    )

print(f"Running on {device}.")
model.eval()



future_coords_offsets_pd = []

timestamps = []

agent_ids = []



with torch.no_grad():

    dataiter = tqdm(test_dataloader)

    

    for data in dataiter:



        inputs = data["image"].to(device)

        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

        targets = data["target_positions"].to(device)



        outputs = model(inputs).reshape(targets.shape)

        

        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())

        timestamps.append(data["timestamp"].numpy().copy())

        agent_ids.append(data["track_id"].numpy().copy())
write_pred_csv('submission.csv',

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd))