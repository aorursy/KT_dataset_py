# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import l5kit

l5kit.__version__
import gc

import os

import pathlib as path

import random

import sys



from tqdm import tqdm

import numpy as np

import pandas as pd

import scipy



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import HTML, display

import cv2
os.listdir('/kaggle/input/lyft-motion-prediction-autonomous-vehicles/')
from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.evaluation import write_pred_csv



os.environ['L5KIT_DATA_FOLDER'] = '/kaggle/input/lyft-motion-prediction-autonomous-vehicles/'

# cfg = load_config_data('/kaggle/input/lyft-config-files/agent_motion_config.yaml')

DEBUG = True
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

        'batch_size': 32,

        'shuffle': True,

        'num_workers': 4

    },

    

    'train_params': {

        'max_num_steps': 12000 if DEBUG else 10000,

        'checkpoint_every_n_steps': 5000,

        

        # 'eval_every_n_steps': -1

    }





}
# print(f'current raster param:\n')

# for k, v in cfg['raster_params'].items():

#     print(f'{k}:{v}')
from l5kit.data import LocalDataManager, ChunkedDataset

dm = LocalDataManager()

dataset_path = dm.require('/kaggle/input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr/')

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
frames = zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)-4000000), desc = 'getting centroid to plot trajectory')):

    frame = zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame['ego_translation'][:2]

sns.set_style('white')

plt.figure(figsize = (13, 8))

ax = sns.scatterplot(coords[:, 0], coords[:, 1], marker = '*', s = 150)

ax.set_xlim([-2500, 1600]);

ax.set_ylim([-2500, 1600]);
len(zarr_dataset.agents)
agent = zarr_dataset.agents[:10000000]



probabilities = agent['label_probabilities']

label_indexes = np.argmax(probabilities, axis = 1)

counts = []

for idx_label, label in enumerate(PERCEPTION_LABELS):

    counts.append(np.sum(label_indexes == idx_label))

    

table = PrettyTable(field_names=['labels', 'counts'])

for count, label in zip(counts, PERCEPTION_LABELS):

    table.add_row([label, count])

print(table)
rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)
data = dataset[1000]



im = dataset.rasterizer.to_rgb(data['image'].T)

sns.set_style('white')

plt.figure(figsize = (10, 7))

plt.imshow(im)
target_positon_pixels = transform_points(data['target_positions']+data['centroid'][:2], data['world_to_image'])

draw_trajectory(cv2.UMat(im), target_positon_pixels, data['target_yaws'], (120, 160, 189))



plt.figure(figsize = (10, 7))

plt.imshow(im)
cfg['raster_params']['map_type'] = 'py_satellite'
sat_rast = build_rasterizer(cfg, dm)

sat_dataset = EgoDataset(cfg, zarr_dataset, sat_rast)
sat_data = sat_dataset[1000]

sat_im = sat_rast.to_rgb(sat_data['image'].T)

plt.figure(figsize = (10, 7))

plt.imshow(sat_im)
sat_target_positon_pixels = transform_points(sat_data['target_positions']+sat_data['centroid'], sat_data['world_to_image'])

draw_trajectory(cv2.UMat(sat_im), sat_target_positon_pixels, sat_data['target_yaws'], TARGET_POINTS_COLOR)

plt.figure(figsize = (10, 7))

plt.imshow(sat_im)
agent_dataset = AgentDataset(cfg, zarr_dataset, rast)
plt.figure(figsize = (10, 7))

agent_data = agent_dataset[1000]

plt.imshow(rast.to_rgb(agent_data['image'].T))
target_positon_pixels = transform_points(agent_data['target_positions']+agent_data['centroid'], agent_data['world_to_image'])

draw_trajectory(cv2.UMat(rast.to_rgb(agent_data['image'].T)), target_positon_pixels, agent_data['target_yaws'], [120, 122, 221])

plt.figure(figsize = (10, 7))

plt.imshow(rast.to_rgb(agent_data['image'].T))
sat_agent_dataset = AgentDataset(cfg, zarr_dataset, sat_rast)

sat_agent_data = sat_agent_dataset[1000]

sat_im = sat_rast.to_rgb(sat_agent_data['image'].T)

plt.figure(figsize = (10, 7))

plt.imshow(sat_im)
import IPython

from IPython.display import display, clear_output

import PIL



scene_idx = 2

indexes = dataset.get_scene_indices(2)

images = []



for idx in indexes:

    data = dataset[idx]

    im = rast.to_rgb(data['image'].T)

    clear_output(wait=True)

    display(PIL.Image.fromarray(im))
images = []

from matplotlib.animation import FuncAnimation

def animate_sol(images):

    def animate(i):

        im.set_data(images[i])

    

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    fig.show()

    return FuncAnimation(fig, animate, frames=len(images), interval=100)



scene_idx = 2

indexes = sat_dataset.get_scene_indices(scene_idx)

images = []

for idx in indexes:

    data = sat_dataset[idx]

    im = rast.to_rgb(data['image'].T)

    clear_output(wait = True)

    images.append(PIL.Image.fromarray(im))

anim = animate_sol(images)

HTML(anim.to_jshtml())
# from torch.utils.data import DataLoader
# dm = LocalDataManager(None)

# train_cfg = cfg['train_data_loader']

# rasterizer = build_rasterizer(cfg, dm)

# train_zarr = ChunkedDataset(dm.require(cfg['train_data_loader']['key'])).open()

# # train_mask = np.load('/kaggle/input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz')['arr_0']

# train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

#                              num_workers=train_cfg["num_workers"])

# print(train_dataset)
# import torch

# import torchvision

# from torchvision import datasets, transforms

# import torch.nn as nn

# import torch.nn.functional as F

# import torch.optim as optim

# from torchvision.models.resnet import resnet18, resnet34
# class Net(nn.Module):

    

#     def __init__(self, cfg):

#         super().__init__()

        

#         self.backbone = resnet18(pretrained=True, progress=True)

        

#         num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

#         num_in_channels = 3 + num_history_channels



#         self.backbone.conv1 = nn.Conv2d(

#             num_in_channels,

#             self.backbone.conv1.out_channels,

#             kernel_size=self.backbone.conv1.kernel_size,

#             stride=self.backbone.conv1.stride,

#             padding=self.backbone.conv1.padding,

#             bias=False,

#         )

        

#         # This is 512 for resnet18 and resnet34;

#         # And it is 2048 for the other resnets

#         backbone_out_features = 512



#         # X, Y coords for the future positions (output shape: Bx50x2)

#         num_targets = 2 * cfg["model_params"]["future_num_frames"]



#         # You can add more layers here.

#         self.head = nn.Sequential(

#             # nn.Dropout(0.2),

#             nn.Linear(in_features=backbone_out_features, out_features=4096),

#         )



#         self.logit = nn.Linear(4096, out_features=num_targets)

        

#     def forward(self, x):

#         x = self.backbone.conv1(x)

#         x = self.backbone.bn1(x)

#         x = self.backbone.relu(x)

#         x = self.backbone.maxpool(x)



#         x = self.backbone.layer1(x)

#         x = self.backbone.layer2(x)

#         x = self.backbone.layer3(x)

#         x = self.backbone.layer4(x)



#         x = self.backbone.avgpool(x)

#         x = torch.flatten(x, 1)

        

#         x = self.head(x)

#         x = self.logit(x)

        

#         return x
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# net = Net(cfg)

# net = net.to(device)

# optimizer = optim.Adam(net.parameters(), lr=1e-3)



# # Later we have to filter the invalid steps.

# criterion = nn.MSELoss(reduction="none")
# tr_it = iter(train_dataloader)
# train_dataloader
# progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

# losses_train = []



# for itr in progress_bar:



#     try:

#         data = next(tr_it)

#     except StopIteration:

#         tr_it = iter(train_dataloader)

#         data = next(tr_it)



#     net.train()

#     torch.set_grad_enabled(True)

    

#     # Forward pass

#     inputs = data["image"].to(device)

#     target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

#     targets = data["target_positions"].to(device)

    

#     outputs = net(inputs).reshape(targets.shape)

#     loss = criterion(outputs, targets)



#     # not all the output steps are valid, but we can filter them out from the loss using availabilities

#     loss = loss * target_availabilities

#     loss = loss.mean()



#     # Backward pass

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()



#     losses_train.append(loss.item())



#     if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:

#         torch.save(model.state_dict(), f'model_state_{itr}.pth')

    

#     progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train[-100:])}")
# torch.save(net.state_dict(), f'model_state_last.pth')
# DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



# SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

# MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"
# cfg = {

#     'format_version': 4,

#     'model_params': {

#         'history_num_frames': 10,

#         'history_step_size': 1,

#         'history_delta_time': 0.1,

#         'future_num_frames': 50,

#         'future_step_size': 1,

#         'future_delta_time': 0.1

#     },

    

#     'raster_params': {

#         'raster_size': [224, 224],

#         'pixel_size': [0.5, 0.5],

#         'ego_center': [0.25, 0.5],

#         'map_type': 'py_semantic',

#         'satellite_map_key': 'aerial_map/aerial_map.png',

#         'semantic_map_key': 'semantic_map/semantic_map.pb',

#         'dataset_meta_key': 'meta.json',

#         'filter_agents_threshold': 0.5

#     },

    

#     'test_data_loader': {

#         'key': 'scenes/test.zarr',

#         'batch_size': 16,

#         'shuffle': False,

#         'num_workers': 4

#     }



# }
# os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

# dm = LocalDataManager(None)
# test_cfg = cfg["test_data_loader"]



# # Rasterizer

# rasterizer = build_rasterizer(cfg, dm)



# # Test dataset/dataloader

# test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

# test_mask = np.load("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")["arr_0"]

# test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

# test_dataloader = DataLoader(test_dataset,

#                              shuffle=test_cfg["shuffle"],

#                              batch_size=test_cfg["batch_size"],

#                              num_workers=test_cfg["num_workers"])
# net.eval()



# future_coords_offsets_pd = []

# timestamps = []

# agent_ids = []



# with torch.no_grad():

#     dataiter = tqdm(test_dataloader )

    

#     for data in dataiter:



#         inputs = data["image"].to(device)

#         target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

#         targets = data["target_positions"].to(device)



#         outputs = net(inputs).reshape(targets.shape)

        

#         future_coords_offsets_pd.append(outputs.cpu().numpy().copy())

#         timestamps.append(data["timestamp"].numpy().copy())

#         agent_ids.append(data["track_id"].numpy().copy())
# device
# write_pred_csv('submission.csv',

#                timestamps=np.concatenate(timestamps),

#                track_ids=np.concatenate(agent_ids),

#                coords=np.concatenate(future_coords_offsets_pd))