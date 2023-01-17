# a brief explanation is found in this video
from IPython.display import HTML


HTML('<center><iframe  width="850" height="450" src="https://www.youtube.com/watch?v=EzylsrXtkxI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp

import gc
import os
from pathlib import Path
import random
import sys
import yaml
from tqdm.notebook import tqdm
import time



!jupyter nbconvert --version
!papermill --version

#!pip install --no-index -f ../input/kaggle-l5kit pip==20.2.2 >/dev/nul
#!pip install --no-index -f ../input/kaggle-l5kit -U l5kit > /dev/nul
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
# TORCH XLA
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
## this script transports l5kit and dependencies
os.system('pip uninstall typing -y')
os.system('pip install --ignore-installed --target=/kaggle/working l5kit')

# Hold back nbconvert to avoid https://github.com/jupyter/nbconvert/issues/1384
os.system('pip install --upgrade --ignore-installed --target=/kaggle/working "nbconvert==5.6.1"')

import l5kit
assert l5kit.__version__ == "1.1.0"

print ('l5kit imported')
print("l5kit version:", l5kit.__version__)
!pip install efficientnet_pytorch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from itertools import islice
from typing import Dict


import torch
from torch import nn, optim


from efficientnet_pytorch import model as enet

from l5kit.evaluation.csv_utils import write_pred_csv
from l5kit.evaluation import compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory


from l5kit.configs import load_config_data


from tqdm import tqdm
from collections import Counter

from prettytable import PrettyTable


from IPython.display import display, clear_output
from IPython.display import HTML

import PIL
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')





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

from torch import nn
from typing import Dict
from pathlib import Path



from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

l5kit_data_folder = "../input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder

IMG_SIZE = 224
# --- Lyft configs ---
cfg = {
          'model_params': {'model_architecture': 'efficientnet-b0',
          'history_num_frames': 0,
          'history_step_size': 1,
          'history_delta_time': 0.1,
          'future_num_frames': 50,
          'future_step_size': 1,
          'future_delta_time': 0.1},

        'raster_params': {'raster_size': [IMG_SIZE, IMG_SIZE],
          'pixel_size': [0.5, 0.5],
          'ego_center': [0.25, 0.5],
          'map_type': 'py_semantic',
          'satellite_map_key': 'aerial_map/aerial_map.png',
          'semantic_map_key': 'semantic_map/semantic_map.pb',
          'dataset_meta_key': 'meta.json',
          'filter_agents_threshold': 0.5},

        'train_data_loader': {'key': 'scenes/train.zarr',
          'batch_size': 4,
          'shuffle': True,
          'num_workers': 0},

        "valid_data_loader":{"key": "scenes/validation.zarr",
                            "batch_size": 4,
                            "shuffle": False,
                            "num_workers": 0},
    
        "sample_data_loader": {
        'key': 'scenes/sample.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 0},
         
        "test_data_loader":{
        'key': "scenes/test.zarr",
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 0},

    
        'train_params': {'checkpoint_every_n_steps': 1000,
          'max_num_steps':3000,
          'eval_every_n_steps': 1000}
        }
print(cfg)
dm = LocalDataManager()
dataset_path = dm.require(cfg["sample_data_loader"]["key"]) # for the EDA we use samples dataset. Smaller and RAM-ready
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


agents = zarr_dataset.agents
probabilities = agents["label_probabilities"]
labels_indexes = np.argmax(probabilities, axis=1)
counts = []
for idx_label, label in enumerate(PERCEPTION_LABELS):
    counts.append(np.sum(labels_indexes == idx_label))
    
table = PrettyTable(field_names=["label", "counts"])
for count, label in zip(counts, PERCEPTION_LABELS):
    table.add_row([label, count])
print(table)
class Config:
    WEIGHT_FILE = "/kaggle/input/lyftpretrained-resnet101/lyft_efficientnetb0.pth" # Model state_dict path of previously trained model
    
    MODEL_NAME = "efficientnet-b0" # b0-b7 could be the different choices.
    
    IMG_SIZE = IMG_SIZE # stated above, together with cfg
    
    PIXEL_SIZE = 0.4
        
    EPOCHS = 2 # Epochs to train the model for.
    VALIDATION = True # A hyperparameter you could use to toggle for validating the model
    l_rate = 1e-4 # Learning rate

    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-5,
        eps=1e-08
    )
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau # Scheduler for learning rate.
    
    criterion = torch.nn.MSELoss(reduction="none") # Loss function.
     
    verbose_steps = 500 # Steps to print model's training status after.
    
config = Config()
def get_dataloader(config, zarr_data, map_type="py_semantic"):
    """Creates DataLoader instance for the given dataset."""
    cfg["raster_params"]["map_type"] = map_type
    rasterizer = build_rasterizer(cfg, dm)
    chunk_data = ChunkedDataset(zarr_data).open()
    agent_data = AgentDataset(cfg, chunk_data, rasterizer)
    dataloader = DataLoader(agent_data, 
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=config["shuffle"]
                           )
    return dataloader
class TPUFitter:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Following some lines are for setting up the AdamW optimizer.
        # See below explanation for details.
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.001},
            
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.l_rate*xm.xrt_world_size())
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        
        # Following function is used for printing to output efficiently. 
        xm.master_print(f'Model Loaded on {self.device}')

    def fit(self, train_loader, validation_loader):
        """Function to fit the model."""
        losses = []
        losses_mean = []
        
        progress = tqdm(range(cfg["train_params"]["max_num_steps"]))
        
        for e in range(config.EPOCHS):
            
            t = time.time() # Get a measurement of time.
            para_loader = pl.ParallelLoader(train_loader, [self.device]) # Distributed loading of the model.
            loss = self.forward(para_loader.per_device_loader(self.device))
            xm.master_print(
                            f'[RESULT]: Train. iter: {e+1}, ' + \
                            f'train_loss: {loss:.5f}, '+ \
                            f' time: {(time.time() - t)/60:.3f}'
                           )
            xm.master_print("\n")
            losses.append(loss.item())
            losses_mean.append(np.mean(losses))
           

    def validation(self, val_loader):
        """Function to validate the model's predictions."""
        val_losses = []
        val_losses_mean = []
        # Setting model to evaluation mode.
        self.model.eval()
        t = time.time()
        
        for step, data in enumerate(val_loader):
            with torch.no_grad():
                inputs = data["image"].to(self.device)
                targets = data["target_positions"].to(self.device)
                target_availabilities = data["target_availabilities"].unsqueeze(-1).to(self.device)
                
                outputs = self.model(inputs)
                outputs = outputs.reshape(targets.shape)
                val_loss = config.criterion(outputs, targets)
                val_loss = val_loss * target_availabilities
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
                val_losses_mean.append(np.mean(val_losses))
                if step % config.verbose_steps == 0:
                    xm.master_print(
                        f'Valid Step {step}, val_loss: ' + \
                        f'{loss:.4f}' + \
                        f' time: {(time.time() - t)/60:.3f}'
                    )                
        return val_losses_mean
         
    def forward(self, train_loader):
        """Function to perform custom forward propagation."""
        
        # Setting model to training mode.
        self.model.train()
        
        t = time.time()
        for step, data in enumerate(train_loader):
            inputs = data["image"].to(self.device)
            target_availabilities = data["target_availabilities"].unsqueeze(-1).to(self.device)
            targets = data["target_positions"].to(self.device)
    
            outputs = self.model(inputs)
            outputs = outputs.reshape(targets.shape)
            loss = config.criterion(outputs, targets)
    
            loss = loss * target_availabilities
            loss = loss.mean()

            if step % config.verbose_steps == 0:
                xm.master_print(
                    f'Train Step {step}, loss: ' + \
                    f'{loss:.4f}' + \
                    f'time: {(time.time() - t)/60:.3f}'
                )
            self.optimizer.zero_grad()
        
            loss.backward()
            xm.optimizer_step(self.optimizer)
        
        self.model.eval()
        self.save('last-checkpoint.bin')
        return loss

    def save(self, path):
        """Function to save the model's current state."""
        xm.save(self.model.state_dict(), path)
# Implementation of class to load the particular EfficientNet model.
class LyftModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = enet.EfficientNet.from_name(config.MODEL_NAME)
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        num_targets = 2*cfg["model_params"]["future_num_frames"]
    
        self.backbone._conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone._conv_stem.out_channels,
            kernel_size=self.backbone._conv_stem.kernel_size,
            stride=self.backbone._conv_stem.stride,
            padding=self.backbone._conv_stem.padding,
            bias=False
        )
    
        self.backbone._fc = nn.Linear(in_features=self.backbone._fc.in_features, out_features=num_targets)
    
    def forward(self, x):
        """Function to perform forward propagation."""
        x = self.backbone(x)
        return x
def get_dataloader(config, zarr_data, subset_len, map_type="py_semantic"):
    """Creates DataLoader instance for the given dataset."""
    
    cfg["raster_params"]["map_type"] = map_type
    rasterizer = build_rasterizer(cfg, dm)
    chunk_data = ChunkedDataset(zarr_data).open()
    agent_data = AgentDataset(cfg, chunk_data, rasterizer)
    
    # Getting Subset of the dataset.
    subset_data = torch.utils.data.Subset(agent_data, range(0, subset_len))
    
    dataloader = DataLoader(subset_data, 
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=config["shuffle"]
                           )
    return dataloader

def train():
    device = xm.xla_device()
    model = LyftModel(cfg).to(device)
    fitter = TPUFitter(model, device)
    
    xm.master_print("Preparing the dataloader..")
    train_dataloader = get_dataloader(cfg["train_data_loader"], dm.require("scenes/train.zarr"), 40000)
    val_dataloader   = get_dataloader(cfg["valid_data_loader"], dm.require("scenes/validate.zarr"), 3000)
    
    xm.master_print("Training the model..")
    fitter.fit(train_dataloader, val_dataloader)
    fitter.save("lyft_model.pth")
    return fitter

model = train()

#print("Saving the model...")
#torch.save(model.state_dict(), "lyft_model.pth")
PATH_TO_DATA = '/kaggle/input/lyft-motion-prediction-autonomous-vehicles/'

test_cfg = cfg["test_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Test dataset/dataloader
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{PATH_TO_DATA}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,
                             shuffle=test_cfg["shuffle"],
                             batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])


print(test_dataset)


def _test():
    

    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []
    device = 'xla:0'
    print(f"device: {device} ready!")
    model = LyftModel(cfg)
    ckpt = torch.load('../input/lyftmodelall/effnet0l2binay_368.bin')
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
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
def forward(data, model, device):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    return outputs

def _test():
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []
    device = 'xla:0'
    print(f"device: {device} ready!")
    model = LyftModel(cfg)
    ckpt = torch.load('./lyft_model.pth')
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        
    # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        agent_ids = []

        progress_bar = tqdm(test_dataloader)
        for data in progress_bar:

        # convert agent coordinates into world offsets
            agents_coords = forward(data, model, device).cpu().numpy().copy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []

        for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):
            coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])
            future_coords_offsets_pd.append(np.stack(coords_offset))
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())


    write_pred_csv(
        "submission.csv",
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(future_coords_offsets_pd),
    )

def _mp_fn(rank, flags):
#     torch.set_default_tensor_type('torch.FloatTensor')
    _test()

FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=1, start_method='fork')