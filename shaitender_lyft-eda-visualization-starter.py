from IPython.display import HTML
HTML('<iframe  width="850" height="450" src="https://www.youtube.com/embed/K0H43N-Hx7w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
### !pip install zarr
!pip install pymap3d==2.1.0
!pip install -U l5kit
import pandas as pd
import numpy as np
import random
import seaborn as sns
import cv2
import os
#import zarr
# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from typing import Dict

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data


import warnings
warnings.filterwarnings("ignore")
DIR_PATH = '../input/lyft-motion-prediction-autonomous-vehicles'
os.listdir(DIR_PATH)
os.environ["L5KIT_DATA_FOLDER"] = DIR_PATH
dm = LocalDataManager(None)

# get config
cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")
rasterizer = build_rasterizer(cfg, dm)

#train
train_dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataloader = DataLoader(AgentDataset(cfg, train_dataset, rasterizer),
                              shuffle=cfg["train_data_loader"]["shuffle"],
                              batch_size=cfg["train_data_loader"]["batch_size"],
                              num_workers=cfg["train_data_loader"]["num_workers"])

# validation
val_dataset = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
val_dataloader = DataLoader(AgentDataset(cfg, val_dataset, rasterizer),
                              shuffle=cfg["val_data_loader"]["shuffle"],
                              batch_size=cfg["val_data_loader"]["batch_size"],
                              num_workers=cfg["val_data_loader"]["num_workers"])




print(train_dataset)
print(" \n ")
print(cfg)
print(" \n ")
print(val_dataset)
class LyftModel(nn.Module):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        self.backbone = resnet18(pretrained=True, progress=True)
        
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
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)
        
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
        
        return x
# ==== INIT MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LyftModel(cfg)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss(reduction="none")
tr_it = iter(train_dataloader)

for itr in range(cfg["train_params"]["max_num_steps"]):

    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)

    model.train()
    torch.set_grad_enabled(True)
    
    # Forward pass
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)

    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

del tr_it
model.eval()
import gc
future_coords_offsets_pd = []
timestamps = []
agent_ids = []

with torch.no_grad():  
    
    for data in val_dataloader:

        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        outputs = model(inputs).reshape(targets.shape)
        
        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
        
        del inputs ,target_availabilities , outputs , targets
        gc.collect()
#submission
from l5kit.evaluation import write_pred_csv
write_pred_csv('submission.csv',
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd))
