!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit


import numpy as np

import os

import torch



from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

from tqdm import tqdm

from typing import Dict

import matplotlib.pyplot as plt





from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer



from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points



from torch.utils.data.dataset import Subset
torch.cuda.is_available()
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = True
cfg = {

    'format_version': 4,

    'model_params': {

        'history_num_frames': 0,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [300, 300], # This is size of the image. That is 300*300 pixel

        'pixel_size': [0.5, 0.5], # One pixel corresponds to these many meters. 

                                    #  So, along the width and height of the image, one pixel will correspond to 0.5 metres in the real world.

        'ego_center': [0.25, 0.5], 

        'map_type': 'py_semantic', # The rasterizer to be used for visualization: 

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 8,

        'shuffle': True,

        'num_workers': 4

    },

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 16,

        'shuffle': False,

        'num_workers': 4

    },

    

    #'train_params': {

     #   'max_num_steps': 100 if DEBUG else 10000,

      #  'checkpoint_every_n_steps': 5000,

        

    'train_params': {"epochs": 10, 'checkpoint_every_n_steps': 200,

          'max_num_steps':1000,

          'eval_every_n_steps': 100

    }



}

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)



cfg["raster_params"]["map_type"] = "py_semantic"

cfg["model_params"]["history_num_frames"] = 10
# Initiate Dataset

train_cfg = cfg["train_data_loader"]

# Rasterizer

rasterizer = build_rasterizer(cfg, dm)

# Test dataset/dataloader

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()



train_dataset = AgentDataset(cfg, train_zarr, rasterizer)



train_dataset = Subset(train_dataset, np.arange(100))



train_dataloader = DataLoader(train_dataset,

                             shuffle=train_cfg["shuffle"],

                             batch_size=train_cfg["batch_size"],

                             num_workers=train_cfg["num_workers"])
test_cfg = cfg["test_data_loader"]



# Rasterizer

# rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)



# test_dataset = Subset(test_dataset, np.arange(100))

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LyftModel(cfg)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
# ==== TRAIN LOOP

tr_it = iter(train_dataloader)

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))



losses_train = []

for epoch in progress_bar:

    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)



    model.train()

    lossees_train = []

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)

    outputs = model(inputs).reshape(targets.shape)

    loss = criterion(outputs, targets)

    loss = loss * target_availabilities

    loss = loss.mean()

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses_train.append(loss.item())

    



    if (epoch + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:

        torch.save(model.state_dict(), f'model_state_{itr}.pth')

        

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    





# Training Analysis

plt.plot(losses_train, label="Training Loss")

plt.xlabel('Training step', fontsize=12) 

plt.ylabel('Loss', fontsize=12)

plt.legend()

plt.show()
if not DEBUG:

    torch.save(model.state_dict(), f'model_state_last.pth')
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