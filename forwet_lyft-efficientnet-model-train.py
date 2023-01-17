# Declaring the path to load efficientNet models.
import sys
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
#IMPORTS

# PyTorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50

# L5kit
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset

# EfficientNet 
from efficientnet_pytorch import model as enet

# Catalyst module
from catalyst import dl
from catalyst.utils import metrics
from catalyst.dl import utils

# Miscellaneous
import os
import gc
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict
import matplotlib.pyplot as plt
from tempfile import gettempdir
from prettytable import PrettyTable
# L5KIT'S CONFIGRATIONS

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager()
cfg = {
        'model_params': {'model_architecture': 'efficientnet-b6',
          'history_num_frames': 0,
          'history_step_size': 1,
          'history_delta_time': 0.1,
          'future_num_frames': 50,
          'future_step_size': 1,
          'future_delta_time': 0.1},

        'raster_params': {'raster_size': [300, 300],
          'pixel_size': [0.33, 0.33],
          'ego_center': [0.25, 0.5],
          'map_type': 'py_semantic',
          'satellite_map_key': 'aerial_map/aerial_map.png',
          'semantic_map_key': 'semantic_map/semantic_map.pb',
          'dataset_meta_key': 'meta.json',
          'filter_agents_threshold': 0.5},

        'train_data_loader': {'key': 'scenes/train.zarr',
          'batch_size': 12,
          'shuffle': True,
          'num_workers': 4},

        "valid_data_loader":{"key": "scenes/validation.zarr",
                            "batch_size": 8,
                            "shuffle": False,
                            "num_workers": 4},
    
        }
def calc_img_size(px_size):
    return int(100/px_size)
# CONFIGRATIONS

WEIGHT_FILE = None # Model state_dict path of previously trained model
MODEL_NAME = "efficientnet-b0"
IMG_SIZE = calc_img_size(cfg["raster_params"]["pixel_size"][0])
VALIDATION = True # A hyperparameter you could use to toggle for validating the model

cfg["raster_params"]["raster_size"] = [IMG_SIZE, IMG_SIZE]
%time
def build_model(cfg) -> torch.nn.Module:
    """Creates an instance of the pretrained model with custom input and output"""
    model = enet.EfficientNet.from_name(MODEL_NAME)
    
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    num_targets = 2*cfg["model_params"]["future_num_frames"]
    
    model._conv_stem = nn.Conv2d(
        num_in_channels,
        model._conv_stem.out_channels,
        kernel_size=model._conv_stem.kernel_size,
        stride=model._conv_stem.stride,
        padding=model._conv_stem.padding,
        bias=False
    )
    
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_targets)
    return model

def forward(data, model, device, criterion):
    """Forward Propogation function"""
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    
    outputs = model(inputs)
    outputs = outputs.reshape(targets.shape)
    loss = criterion(outputs, targets)
    
    loss = loss * target_availabilities
    loss = loss.mean()

    # Disabling rmse loss pertaining the mse loss.
    # loss = torch.sqrt(loss) # Using RMSE loss
    return loss, outputs


def get_dataloader(config, zarr_data, subset_len, map_type="py_semantic"):
    """Creates DataLoader instance for the given dataset."""
    cfg["raster_params"]["map_type"] = map_type
    rasterizer = build_rasterizer(cfg, dm)
    chunk_data = ChunkedDataset(zarr_data).open()
    agent_data = AgentDataset(cfg, chunk_data, rasterizer)
    
    # Sample the dataset
    subset_data = torch.utils.data.Subset(agent_data, range(0, subset_len))
    
    dataloader = DataLoader(subset_data, 
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=config["shuffle"]
                           )
    return dataloader
%time
def train(opt=None, criterion=None, lrate=1e-2):
        """Function for training the model"""
        print("Building Model...")
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lrate) if opt is None else opt
        criterion = nn.MSELoss(reduction="none")
        
        if WEIGHT_FILE is not None:
            state_dict = torch.load(WEIGHT_FILE, map_location=device)
            model.load_state_dict(state_dict)
        
        print("Prepairing Dataloader...")
        train_dataloader = get_dataloader(cfg["train_data_loader"], dm.require("scenes/train.zarr"), 12000)
        
        if VALIDATION:
            valid_dataloader = get_dataloader(cfg["valid_data_loader"], dm.require("scenes/validate.zarr"), 500)
            
        print("Training...")
        loaders = {
                    "train": train_dataloader,
                    "valid": valid_dataloader
                }

        device = utils.get_device()
        runner = LyftRunner(device=device)
        
        runner.train(
                model=model,
                optimizer=optimizer,
                loaders=loaders,
                logdir="./logs",
                num_epochs=5,
                verbose=True,
                load_best_on_end=True
            )
        return model

class LyftRunner(dl.Runner):

    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        x, y = batch['image'], batch['target_positions']
        y_hat = self.model(x).reshape(y.shape)
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        criterion = torch.nn.MSELoss(reduction="none")
        loss = criterion(y_hat, y)
        loss = loss * target_availabilities
        loss = loss.mean()
        self.batch_metrics.update(
            {"loss": loss}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
# Testing pixel_size and raster_size START 
# Preparing the EgoDataset from sample zarr file.
sample_zarr = dm.require("scenes/sample.zarr")
sample_chunk = ChunkedDataset(sample_zarr).open()
rasterizer = build_rasterizer(cfg, dm)
sample_ego = EgoDataset(cfg, sample_chunk, rasterizer)
fig, ax = plt.subplots(3, 3, figsize=(15,15))
ax = ax.flatten()
for i in range(9):
    idx = np.random.randint(500)
    data = sample_ego[idx]
    im = data["image"].transpose(1, 2, 0)
    im = sample_ego.rasterizer.to_rgb(im)
    data_positions = transform_points(data["target_positions"]+data["centroid"], data["world_to_image"])
    draw_trajectory(im, data_positions, data["target_yaws"], TARGET_POINTS_COLOR)
    ax[i].imshow(im[::-1])
plt.show()
# Testing END
# Use the above defined utility script to train the model.
model = train()
# Saving model on final iteration
torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
