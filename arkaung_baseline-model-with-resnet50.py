!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit
from typing import Dict



from tempfile import gettempdir

import matplotlib.pyplot as plt

import numpy as np

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

from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS

from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

from l5kit.geometry import transform_points

from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

from prettytable import PrettyTable

from pathlib import Path



import os
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

dm = LocalDataManager(None)

# get config

cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")
def build_model(cfg: Dict) -> torch.nn.Module:

    # load pre-trained Conv2D model

    model = resnet50(pretrained=True)



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

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)



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
cfg['train_data_loader']['key'] = 'scenes/train.zarr'

cfg['train_data_loader']['num_workers'] = 8

cfg['train_data_loader']['batch_size'] = 64
# ===== INIT DATASET

train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

                             num_workers=train_cfg["num_workers"])

print(train_dataset)
# ==== INIT MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = build_model(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
cfg["train_params"]["max_num_steps"] = 10
# ==== TRAIN LOOP

tr_it = iter(train_dataloader)

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

losses_train = []

for _ in progress_bar:

    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)

    model.train()

    torch.set_grad_enabled(True)

    loss, _ = forward(data, model, device, criterion)



    # Backward pass

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
# cfg['val_data_loader']['key'] = 'scenes/validate.zarr'
# # Loading validation dataset

# eval_cfg = cfg["val_data_loader"]

# rasterizer = build_rasterizer(cfg, dm)

# eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()

# eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer)

# eval_dataloader = DataLoader(eval_dataset, 

#                              shuffle=eval_cfg["shuffle"], 

#                              batch_size=eval_cfg["batch_size"], 

#                              num_workers=eval_cfg["num_workers"])

# print(eval_dataset)
# # ==== EVAL LOOP

# model.eval()

# torch.set_grad_enabled(False)



# # store information for evaluation

# future_coords_offsets_pd = []

# timestamps = []



# agent_ids = []

# progress_bar = tqdm(eval_dataloader)

# for data in progress_bar:

#     _, ouputs = forward(data, model, device, criterion)

#     future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())

#     timestamps.append(data["timestamp"].numpy().copy())

#     agent_ids.append(data["track_id"].numpy().copy())
# error = compute_error_csv(eval_gt_path, pred_path)

# print(f"NLL: {error:.5f}\nL2: {np.sqrt(2*error/cfg['model_params']['future_num_frames']):.5f}")