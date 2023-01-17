!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit

import matplotlib.pyplot as plt



import numpy as np



from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset



from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

import os

PATH_TO_DATA = "../input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = PATH_TO_DATA
cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

        

        'history_num_frames': 0,

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

    

    'val_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 12,

        'shuffle': False,

        'num_workers': 4

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },

    

    'train_params': {

        'checkpoint_every_n_steps': 5000,

        'max_num_steps': 1000,

        

    }

}

zarr_loc = cfg["train_data_loader"]["key"]



dm = LocalDataManager()

train_zarr = ChunkedDataset(dm.require(zarr_loc)).open()



#let's see what one of the objects looks like

print(train_zarr)
#Let us visualize the EGO DATASET
rast = build_rasterizer(cfg, dm)

#Rasterisation is one of the typical techniques of rendering 3D models

dataset = EgoDataset(cfg, train_zarr , rast)
#For visualizing the Av. Av stands for Autonomous Vehicle

#Let's get a sample from the dataset and use our rasterizer to get an RGB image we can plot
data = dataset[70]



im = data['image'].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()

# FROM WHAT I CAN DEDUCE THE RED BOX YOU SEE IS THE intersection entry wait line for ego car

#If it is Yellow means an intersection entry wait line for other cars while Green is intersection exit line. 
#Let Visualize other images|
data = dataset[30]



im = data['image'].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
#Let me try visualizing image through another rasterizer
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, train_zarr, rast)

data = dataset[2]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
#There are two ways in which our self driving car could go it can either continue going,

# straight or it can make a turn to the right hand side
#Visualizing an agent

dataset = AgentDataset(cfg, train_zarr, rast)

data = dataset[0]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
from IPython.display import display, clear_output

from IPython.display import HTML



import PIL

import matplotlib.pyplot as plt

from matplotlib import animation, rc

def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, train_zarr, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

anim = animate_solution(images)

HTML(anim.to_jshtml())
#STILL WORKING ON THIS NOTEBOOK. NOTEBOOK YET TO BE COMPLETED PLEASE STAY TUNE FOR MORE 

#THANK YOU
from IPython.display import display, clear_output

from IPython.display import HTML



import PIL

import matplotlib.pyplot as plt

from matplotlib import animation, rc

def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, train_zarr, rast)

scene_idx = 60

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

anim = animate_solution(images)

HTML(anim.to_jshtml())
#Loading the train dataset which for me is the train_zarr

#we will load the train_zarr into a chunkdataset object.

#The object has a reference to the different arrays into the zarr e.g the agent and the traffic light

#we will pass the agent dataset into the torchdataLoader
from torch.utils.data import DataLoader

train_cfg = cfg['train_data_loader']

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg['key'])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg['shuffle'], batch_size=train_cfg['batch_size'], 

                             num_workers=train_cfg['num_workers'])

print(train_dataset)
from torch import nn

from torchvision.models.resnet import resnet18

from typing import Dict



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
#I WILL INITIALIZE THE MODEL HERE AND WE WILL USE THE ADAM OPTIMIZER

#WE WILL USE THE ADAM OPTIMIZER BECAUSE IT IS GOOD FOR LARGE DATASET
from torch import optim

from tqdm import tqdm



import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = LyftModel(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
#TRAIN LOOP
tr_it = iter(train_dataloader)

progress_bar = tqdm(range(cfg['train_params']['max_num_steps']))

losses_train = []

for _ in progress_bar:

    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)

        

    model.train()

    torch.set_grad_enabled(True)

    #THE INPUTS IMAGES AND THE TARGET POSITIONS     

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)

    loss = criterion(outputs, targets)

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

#STILL WORKING ON THIS NOTEBOOK. WILL BE COMPLETED SOON, STAY TUNED. THANKS FOR VIEWING 