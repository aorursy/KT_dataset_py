!pip install --upgrade pip
!pip install pymap3d==2.1.0
!pip install -U l5kit
 !pip3 install resnet_pytorch
from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os
from torchvision.models.resnet import resnet18
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager(None)
# get config
#cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")
#MODEL_NAME = "wide_resnet18"
IMG_SIZE = 224
# --- Lyft configs ---
cfg = {
          'model_params': {'model_architecture': 'resnet18',
          'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet101_output",
        'lr': 1e-4,
        'weight_path': "/kaggle/input/lyftpretrained-resnet101/lyft_resnet101_model.pth",
        'train': True,
        'predict': True},

        'raster_params': {'raster_size': [IMG_SIZE, IMG_SIZE],
          'pixel_size': [0.5, 0.5],
          'ego_center': [0.25, 0.5],
          'map_type': 'py_semantic',
          'satellite_map_key': 'aerial_map/aerial_map.png',
          'semantic_map_key': 'semantic_map/semantic_map.pb',
          'dataset_meta_key': 'meta.json',
          'filter_agents_threshold': 0.5},

        'train_data_loader': {'key': 'scenes/train.zarr',
          'batch_size': 8,
          'shuffle': True,
          'num_workers': 0},

        "valid_data_loader":{"key": "scenes/validate.zarr",
                            "batch_size": 8,
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

    
        'train_params': {"epochs": 10, 'checkpoint_every_n_steps': 200,
          'max_num_steps':1000,
          'eval_every_n_steps': 100}
        }
print(cfg)
def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet18(pretrained=True)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=512, out_features=num_targets)

    return model
def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs
# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
print(train_dataset)
# ===== INIT VALIDATION DATASET============================================================
valid_cfg = cfg["valid_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
validate_zarr = ChunkedDataset(dm.require(valid_cfg["key"])).open()
valid_dataset = AgentDataset(cfg, validate_zarr, rasterizer)
valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"], 
                             num_workers=valid_cfg["num_workers"])
print("==================================VALIDATION DATA==================================")
print(valid_dataset)
# ==== INIT MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction="none")
VALIDATION = True
def train(train_dataloader, valid_dataloader, opt=None, criterion=None, lrate=1e-4):
        """Function for training the model"""
        print("Building Model...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss(reduction="none")
                             
                
        print("Training...")
        losses = []
        losses_mean = []
        
        val_losses = []
        val_losses_mean = []
        
        progress = tqdm(range(cfg["train_params"]["max_num_steps"]))
        
        train_iter = iter(train_dataloader)
        val_iter = iter(valid_dataloader)
        
        for i in progress:
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                data = next(train_iter)
                    
            model.train()
            torch.set_grad_enabled(True)
                    
            loss, _ = forward(data, model, device, criterion)
                        
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if VALIDATION:
                with torch.no_grad():
                    try:
                        val_data = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dataloader)
                        val_data = next(val_iter)

                    val_loss, _  = forward(val_data, model, device, criterion)
                    val_losses.append(val_loss.item())
                    val_losses_mean.append(np.mean(val_losses))
                    
                desc = f"Loss: {round(loss.item(), 4)} Validation Loss: {round(val_loss.item(), 4)}"
            else:
                desc = f"Loss: {round(loss.item(), 4)}"
                
            #if len(losses)>0 and loss < min(losses):
            #    print(f"Loss improved from {min(losses)} to {loss}")
                
            
            
            losses.append(loss.item())
            losses_mean.append(np.mean(losses))
            progress.set_description(desc)
            
        return losses_mean, val_losses_mean, model
losses, val_losses, model = train(train_dataloader, valid_dataloader)
# Training Analysis
plt.plot(losses, c="red", label="Mean Training Loss")
plt.plot(val_losses, c="green", label="Mean Validation Loss")
plt.xlabel('Training step', fontsize=12) 
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.show()
# # Loading eval dataset
eval_cfg = cfg["sample_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer)
eval_dataloader = DataLoader(eval_dataset, 
                             shuffle=eval_cfg["shuffle"], 
                             batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)
#==== EVAL LOOP
model.eval()
torch.set_grad_enabled(False)
# store information for evaluation
future_coords_offsets_pd = []
timestamps = []

agent_ids = []
progress_bar = tqdm(eval_dataloader)
for data in progress_bar:
    _, ouputs = forward(data, model, device, criterion)
    future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())
error = compute_error_csv(eval_gt_path, pred_path)
print(f"NLL: {error:.5f}\nL2: {np.sqrt(2*error/cfg['model_params']['future_num_frames']):.5f}")