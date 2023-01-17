import numpy as np

import os

import torch

import random

import cv2



from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

from tqdm import tqdm

from typing import Dict

from typing import Tuple



import matplotlib.pyplot as plt



# Add this notebook output as utility script instead of pip installing it:

# https://www.kaggle.com/philculliton/kaggle-l5kit. Search this by "philculliton/kaggle-l5kit".

from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer



# Seed everything

torch.manual_seed(28)

torch.cuda.manual_seed(28)

np.random.seed(28)

random.seed(28)
BASE_DIR = '/kaggle/input/lyft-motion-prediction-autonomous-vehicles'

os.environ['L5KIT_DATA_FOLDER'] = BASE_DIR



config = {

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

        'key': 'scenes/sample.zarr',

        'batch_size': 20,

        'shuffle': False,

        'num_workers': 0

    },

    

    'train_params': {

        'max_num_steps': 100,

        'checkpoint_every_n_steps': 5000

    }

}
# Initialize local data manager

data_manager = LocalDataManager()



train_config = config['train_data_loader']



# Train dataset/dataloader

train_zarr = ChunkedDataset(data_manager.require(train_config['key'])).open()





def load_dataset():

    # Build Rasterizer

    rasterizer = build_rasterizer(config, data_manager)

    

    train_dataset = AgentDataset(config, train_zarr, rasterizer)

    return train_dataset[100]
data = load_dataset()

# batch = next(iter(train_dataloader))

print('List of available features:\n\n{}'.format('\n'.join(data.keys())))
f, ax = plt.subplots(5, 5, figsize=(20, 20))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(data['image'][i], cmap='Greys')

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)
config['raster_params']['pixel_size'] = [0.3, 0.3]

data = load_dataset()



f, ax = plt.subplots(5, 5, figsize=(20, 20))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(data['image'][i], cmap='Greys')

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)



# Revert back the pixel_size

config['raster_params']['pixel_size'] = [0.5, 0.5]
sizes = [[150, 150], [224, 224], [250, 250], [350, 350], [450, 450], [500, 500]]



f, ax = plt.subplots(2, 3, figsize=(20, 12))

ax = ax.flatten()



for i in range(6):

    config['raster_params']['raster_size'] = sizes[i]

    data = load_dataset()

    

    ax[i].imshow(data['image'][-3:].transpose(1, 2, 0), cmap='Greys')

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)



# Revert back the pixel_size

config['raster_params']['raster_size'] = [224, 224]
config['raster_params']['ego_center'] = [0.5, 0.5]

data = load_dataset()



f, ax = plt.subplots(5, 5, figsize=(20, 20))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(data['image'][i], cmap='Greys')



# Revert back the ego_center

config['raster_params']['ego_center'] = [0.25, 0.5]
plt.figure(figsize=(8, 8))

plt.imshow(data['image'][-3:].transpose(1, 2, 0))

plt.show()