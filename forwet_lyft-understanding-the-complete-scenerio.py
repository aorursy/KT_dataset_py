# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
# UTILITY FUNCTIONS

def visualize_image(ds, idx, axis=None):
    data = ds[idx]
    im = data["image"].transpose(1, 2, 0)
    im = ds.rasterizer.to_rgb(im)
    if axis:
        axis.imshow(im[::-1])
    else:
        plt.imshow(im[::-1])

def build_model(cfg: Dict) -> torch.nn.Module:
    # Backbone model
    model = resnet50(pretrained=True)
    
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    
    # Getting input channels for first layer of the model.
    num_in_channels = 3 + num_history_channels
    
    # Getting the number of targets.
    num_targets = 2*cfg["model_params"]["future_num_frames"]
    
    # Adjusting the first layer for the number of input channels.
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    
    # Adjusting the final layer for the number of targets.
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)
    return model

def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    targets = data["target_positions"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    
    outputs = model(inputs).reshape(targets.shape).to(device)
    
    loss = torch.sqrt(criterion(outputs, targets))
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs
dm = LocalDataManager()
cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")
sample_zarr = dm.require("scenes/sample.zarr")
sample_chunk = ChunkedDataset(sample_zarr).open()
# Let us have a look at the above discussed attributes of ChunkedDataset instance.
print(sample_chunk.scenes)
print()
print(sample_chunk.agents)
print()
print(sample_chunk.tl_faces)
print()
print(sample_chunk.frames)
# Look at the cfg
cfg
# Semantic Rasterizer
cfg["raster_params"]["map_type"] = "py_semantic"
raster_sem = build_rasterizer(cfg, dm)

# Satellite Rasterizer
cfg["raster_params"]["map_type"] = "py_satellite"
raster_sat = build_rasterizer(cfg, dm)
cfg["raster_params"]["map_type"] = "py_semantic"
ego_dataset_sem = EgoDataset(cfg, sample_chunk, raster_sem)

cfg["raster_params"]["map_type"] = "py_satellite"
ego_dataset_sat = EgoDataset(cfg, sample_chunk, raster_sat)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()
for i in range(3):
    visualize_image(ego_dataset_sem, i+10, axis=ax[i])
    ax[i].set_title("Semantic Image")
    
    visualize_image(ego_dataset_sat, i+10, axis=ax[i+3])
    ax[i+3].set_title("Satellite Image")
    
cfg = {
    'format_version': 4,
 
    'model_params': {'model_architecture': 'resnet50',
    'history_num_frames': 0,
    'history_step_size': 1,
    'history_delta_time': 0.1,
    'future_num_frames': 50,
    'future_step_size': 1,
    'future_delta_time': 0.1},
 
    'raster_params': {'raster_size': [224, 224],
    'pixel_size': [0.5, 0.5],
    'ego_center': [0.25, 0.5],
    'map_type': 'py_semantic',
    'satellite_map_key': 'aerial_map/aerial_map.png',
    'semantic_map_key': 'semantic_map/semantic_map.pb',
    'dataset_meta_key': 'meta.json',
    'filter_agents_threshold': 0.5},
 
    'train_data_loader': {'key': 'scenes/sample.zarr',
    'batch_size': 12,
    'shuffle': True,
    'num_workers': 16},
 
    'test_data_loader': {'key': 'scenes/test.zarr',
    'batch_size': 8,
    'shuffle': False,
    'num_workers': 8},

    'train_params': {'checkpoint_every_n_steps': 10000,
    'max_num_steps': 5,
    'eval_every_n_steps': 10000}}
sample_agent = AgentDataset(cfg, sample_chunk, raster_sem)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Generating Dataloader object for out sample_agent dataset
sample_config = cfg["train_data_loader"]
sample_dataloader = DataLoader(sample_agent, batch_size=sample_config["batch_size"],
                               shuffle=sample_config["shuffle"], num_workers=sample_config["num_workers"])
## TRAINING LOOP
iter_train = iter(sample_dataloader)
progress = tqdm(range(cfg["train_params"]["max_num_steps"]))

for _ in progress:
    try:
        train_data = next(iter_train)
    except:
        iter_train = iter(sample_dataloader)
        train_data = next(iter_train)
    
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = forward(train_data, model, device, criterion)
    
    # Zeroing out gradients
    optimizer.zero_grad()
    
    # Backprop
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    progress.set_description(f"Loss: {loss.item()}")
# Preparing Test Data
test_config = cfg["test_data_loader"]
test_zarr = dm.require("scenes/test.zarr")
test_chunk  = ChunkedDataset(test_zarr).open()
test_mask  = np.load("../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")["arr_0"]
test_agent = AgentDataset(cfg, test_chunk, raster_sem, agents_mask = test_mask)
test_dataloader = DataLoader(test_agent, batch_size=test_config["batch_size"], shuffle=test_config["shuffle"],
                            num_workers=test_config["num_workers"])
model.eval()

future_coords = []
timestamps = []
track_ids = []

with torch.no_grad():
    iter_test = tqdm(test_dataloader)
    for data in iter_test:
        _, outputs = forward(data, model, device, criterion)
        timestamps.append(data["timestamp"])
        track_ids.append(data["track_id"])
write_pred_csv("submission.csv",
              timestamps=np.concatenate(timestamps),
              track_ids=np.concatenate(track_ids),
              coords=np.concatenate(future_coords))