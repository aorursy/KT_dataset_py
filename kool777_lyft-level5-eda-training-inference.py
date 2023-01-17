from IPython.display import HTML



HTML('<center><iframe  width="850" height="450" src="https://www.youtube.com/embed/K0H43N-Hx7w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
# import packages

import os, gc

import zarr

import numpy as np 

import pandas as pd 

from tqdm import tqdm

from typing import Dict

from collections import Counter

from prettytable import PrettyTable



#level5 toolkit

from l5kit.data import PERCEPTION_LABELS

from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.data import ChunkedDataset, LocalDataManager



# level5 toolkit 

from l5kit.configs import load_config_data

from l5kit.geometry import transform_points

from l5kit.rasterization import build_rasterizer

from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR

from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import animation

from colorama import Fore, Back, Style



# deep learning

import torch

from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18, resnet50, resnet34



# check files in directory

print((os.listdir('../input/lyft-motion-prediction-autonomous-vehicles/')))



plt.rc('animation', html='jshtml')



%matplotlib inline
# animation for scene

def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=80)
import numpy as np

my_arr = np.zeros(3, dtype=[("color", (np.uint8, 3)), ("label", np.bool)])



print(my_arr[0])
my_arr[0]["color"] = [0, 218, 130]

my_arr[0]["label"] = True

my_arr[1]["color"] = [245, 59, 255]

my_arr[1]["label"] = True

my_arr[1]["color"] = [7, 6, 97]

my_arr[1]["label"] = True



print(my_arr)
train = zarr.open("../input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr")

validation = zarr.open("../input/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr")

test = zarr.open("../input/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr/")

train.info
print(f'We have {len(train.agents)} agents, {len(train.scenes)} scenes, {len(train.frames)} frames and {len(train.traffic_light_faces)} traffic light faces in train.zarr.')

print(f'We have {len(validation.agents)} agents, {len(validation.scenes)} scenes, {len(validation.frames)} frames and {len(validation.traffic_light_faces)} traffic light faces in validation.zarr.')

print(f'We have {len(test.agents)} agents, {len(test.scenes)} scenes, {len(test.frames)} frames and {len(test.traffic_light_faces)} traffic light faces in test.zarr.')

# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"



# get configuration yaml

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")

print(cfg)
# Raster Parameters

print(f'current raster_param:\n')

for k,v in cfg["raster_params"].items():

    print(f"{k}:{v}")
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
print(dataset_path)
agents = pd.DataFrame.from_records(zarr_dataset.agents, columns = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities'])

agents.head()
agents[['centroid_x','centroid_y']] = agents['centroid'].to_list()

agents = agents.drop('centroid', axis=1)

agents_new = agents[["centroid_x", "centroid_y", "extent", "yaw", "velocity", "track_id", "label_probabilities"]]

del agents

agents_new
fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.scatter(agents_new['centroid_x'], agents_new['centroid_y'], marker='+')

plt.xlabel('x', fontsize=11); plt.ylabel('y', fontsize=11)

plt.title("Centroids distribution (sample.zarr)")

plt.show()
agents_new[['extent_x','extent_y', 'extent_z']] = agents_new['extent'].to_list()

agents_new = agents_new.drop('extent', axis=1)

agents = agents_new[["centroid_x", "centroid_y", 'extent_x', 'extent_y', 'extent_z', "yaw", "velocity", "track_id", "label_probabilities"]]

del agents_new

agents
sns.axes_style("white")



fig, ax = plt.subplots(1,3,figsize=(16,5))



plt.subplot(1,3,1)

sns.kdeplot(agents['extent_x'], shade=True, color='red');

plt.title("Extent_x distribution")



plt.subplot(1,3,2)

sns.kdeplot(agents['extent_y'], shade=True, color='steelblue');

plt.title("Extent_y distribution")



plt.subplot(1,3,3)

sns.kdeplot(agents['extent_z'], shade=True, color='green');

plt.title("Extent_z distribution")



plt.show();
sns.set_style('whitegrid')



fig, ax = plt.subplots(1,3,figsize=(16,5))

plt.subplot(1,3,1)

plt.scatter(agents['extent_x'], agents['extent_y'], marker='*')

plt.xlabel('ex', fontsize=11); plt.ylabel('ey', fontsize=11)

plt.title("Extent: ex-ey")



plt.subplot(1,3,2)

plt.scatter(agents['extent_y'], agents['extent_z'], marker='*', color="red")

plt.xlabel('ey', fontsize=11); plt.ylabel('ez', fontsize=11)

plt.title("Extent: ey-ez")



plt.subplot(1,3,3)

plt.scatter(agents['extent_z'], agents['extent_x'], marker='*', color="green")

plt.xlabel('ez', fontsize=11); plt.ylabel('ex', fontsize=11)

plt.title("Extent: ez-ex")



plt.show();
fig, ax = plt.subplots(1,1,figsize=(10,8))

sns.distplot(agents['yaw'])

plt.title("Yaw Distribution")

plt.show()
agents[['velocity_x','velocity_y']] = agents['velocity'].to_list()

agents_vel = agents.drop('velocity', axis=1)

agents_v = agents_vel[["centroid_x", "centroid_y", 'extent_x', 'extent_y', 'extent_z', "yaw", "velocity_x", "velocity_y", "track_id", "label_probabilities"]]

del agents

agents_v
fig, ax = plt.subplots(1,1,figsize=(10,8))



with sns.axes_style("whitegrid"):

    sns.scatterplot(x=agents_v["velocity_x"], y=agents_v["velocity_y"], color='k');

    plt.title('Velocity Distribution')
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
print(f'{Fore.YELLOW}Total number of agents in sample .zarr files is {Style.RESET_ALL}{len(zarr_dataset.agents)}. {Fore.BLUE}\nAfter summing up the elements in count column we can see we have {Style.RESET_ALL}{(1324481 + 519385 + 6688 + 43182)} {Fore.BLUE}agents in total.')
scenes = pd.DataFrame.from_records(zarr_dataset.scenes, columns = ['frame_index_interval', 'host', 'start_time', 'end_time'])

scenes.head()
scenes[['frame_start_index','frame_end_index']] = scenes['frame_index_interval'].to_list()

scenes_new = scenes.drop('frame_index_interval', axis=1)

scenes_new = scenes_new[["frame_start_index", "frame_end_index", 'host', 'start_time', 'end_time']]

del scenes

scenes_new.head()
f = plt.figure(figsize=(10, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0,0])

    sns.scatterplot(scenes_new['frame_start_index'], scenes_new['frame_end_index'])

    plt.title('Frame Index Interval Distribution')

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0,1])

    sns.scatterplot(scenes_new['frame_start_index'], scenes_new['frame_end_index'], hue=scenes_new['host'])

    plt.title('Frame Index Interval Distribution (Grouped per host)')

    

f.tight_layout()
f = plt.figure(figsize=(10, 8))



with sns.axes_style("white"):

    sns.countplot(scenes_new['host']);

    plt.title("Host Count")
frames = pd.DataFrame.from_records(zarr_dataset.frames, columns = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 'ego_translation','ego_rotation'])

frames.head()
frames[['ego_translation_x', 'ego_translation_y', 'ego_translation_z']] = frames['ego_translation'].to_list()

frames_new = frames.drop('ego_translation', axis=1)

frames_new = frames_new[['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval',

                         'ego_translation_x', 'ego_translation_y', 'ego_translation_z', 'ego_rotation']]

del frames

frames_new.head()
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 3)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0,0])

    sns.distplot(frames_new['ego_translation_x'], color='Orange')

    plt.title('Ego Translation Distribution X')

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0,1])

    sns.distplot(frames_new['ego_translation_y'], color='Red')

    plt.title('Ego Translation Distribution Y')

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0,2])

    sns.distplot(frames_new['ego_translation_z'], color='Green')

    plt.title('Ego Translation Distribution Z')

    

f.tight_layout()
f = plt.figure(figsize=(16, 6))

gs = f.add_gridspec(1, 3)



with sns.axes_style("darkgrid"):

    ax = f.add_subplot(gs[0,0])

    plt.scatter(frames_new['ego_translation_x'], frames_new['ego_translation_y'],

                    color='darkkhaki', marker='+')

    plt.title('Ego Translation X-Y')

    plt.xlabel('ego_translation_x')

    plt.ylabel('ego_translation_y')

    

with sns.axes_style("darkgrid"):

    ax = f.add_subplot(gs[0,1])

    plt.scatter(frames_new['ego_translation_y'], frames_new['ego_translation_z'],

                    color='slateblue', marker='*')

    plt.title('Ego Translation Distribution Y-Z')

    plt.xlabel('ego_translation_y')

    plt.ylabel('ego_translation_z')

    

with sns.axes_style("darkgrid"):

    ax = f.add_subplot(gs[0,2])

    plt.scatter(frames_new['ego_translation_z'], frames_new['ego_translation_x'],

                    color='turquoise', marker='^')

    plt.title('Ego Translation Distribution Z-X')

    plt.xlabel('ego_translation_z')

    plt.ylabel('ego_translation_x')

    

f.tight_layout()
fig, ax = plt.subplots(3,3,figsize=(16,16))

colors = ['red', 'blue', 'green', 'magenta', 'orange', 'darkblue', 'black', 'cyan', 'darkgreen']

for i in range(0,3):

    for j in range(0,3):

        df = frames_new['ego_rotation'].apply(lambda x: x[i][j])

        plt.subplot(3,3,i * 3 + j + 1)

        sns.distplot(df, hist=False, color = colors[ i * 3 + j  ])

        plt.xlabel(f'r[ {i + 1} ][ {j + 1} ]')

fig.suptitle("Ego rotation angles distribution", size=14)

plt.show()
traffic_light_faces = pd.DataFrame.from_records(zarr_dataset.tl_faces, columns = ['face_id', 'traffic_light_id', 'traffic_light_face_status'])

traffic_light_faces.head()
sns.set_style({'axes.grid': False})



def visualize_rgb_image(dataset, index, title="", ax=None):

    """Visualizes Rasterizer's RGB image"""

    data = dataset[index]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)



    if ax is None:

        fig, ax = plt.subplots()

    if title:

        ax.set_title(title)

    ax.imshow(im[::-1])

# Prepare all rasterizer and EgoDataset for each rasterizer

rasterizer_dict = {}

dataset_dict = {}



rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]



for i, key in enumerate(rasterizer_type_list):

    # print("key", key)

    cfg["raster_params"]["map_type"] = key

    rasterizer_dict[key] = build_rasterizer(cfg, dm)

    dataset_dict[key] = EgoDataset(cfg, zarr_dataset, rasterizer_dict[key])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, key in enumerate(["stub_debug", "satellite_debug", "semantic_debug", "box_debug", "py_satellite", "py_semantic"]):

    visualize_rgb_image(dataset_dict[key], index=0, title=f"{key}: {type(rasterizer_dict[key]).__name__}", ax=axes[i])

fig.show()
cfg["raster_params"]["map_type"] = "py_semantic"



# raster object for visualization

rast = build_rasterizer(cfg, dm)



# EgoDataset object

dataset = EgoDataset(cfg, zarr_dataset, rast)



# select one example from our dataset

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])



# plot ground truth trajectory

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.rcParams['figure.figsize'] = 10, 10

plt.title('Ground Truth Trajectory of Autonomous Vehicle',fontsize=15)

plt.imshow(im[::-1])

plt.show()
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)



# EgoDataset object

dataset = EgoDataset(cfg, zarr_dataset, rast)

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.rcParams['figure.figsize'] = 10, 10

plt.title('Satellite View: Ground Truth Trajectory of Autonomous Vehicle',fontsize=15)

plt.imshow(im[::-1])

plt.show()
# AgentDataset object

dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.rcParams['figure.figsize'] = 10, 10

plt.title('Agent',fontsize=15)

plt.imshow(im[::-1])

plt.show()
from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

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

    

# animation    

anim = animate_solution(images)

HTML(anim.to_jshtml())
# satellite view

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

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

    

# animation    

anim = animate_solution(images)

HTML(anim.to_jshtml())
gc.collect()
DEBUG = True



# training cfg

training_cfg = {

    

    'format_version': 4,

    

     ## Model options

    'model_params': {

        'model_architecture': 'resnet34',

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1,

    },



    ## Input raster parameters

    'raster_params': {

        

        'raster_size': [224, 224], # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.

        'pixel_size': [0.5, 0.5], # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.

        'ego_center': [0.25, 0.5],

        'map_type': "py_semantic",

        

        # the keys are relative to the dataset environment variable

        'satellite_map_key': "aerial_map/aerial_map.png",

        'semantic_map_key': "semantic_map/semantic_map.pb",

        'dataset_meta_key': "meta.json",



        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being

        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.

        'filter_agents_threshold': 0.5

    },



    ## Data loader options

    'train_data_loader': {

        'key': "scenes/train.zarr",

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 4

    },



    ## Train params

    'train_params': {

        'checkpoint_every_n_steps': 5000,

        'max_num_steps': 100 if DEBUG else 10000

    }

}



# inference cfg

inference_cfg = {

    

    'format_version': 4,

    'model_params': {

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [300, 300],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

        'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    }



}
# root directory

DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



#submission

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"



# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)

print(training_cfg)
# training cfg

train_cfg = training_cfg["train_data_loader"]



# rasterizer

rasterizer = build_rasterizer(training_cfg, dm)



# dataloader

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(training_cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

                             num_workers=train_cfg["num_workers"])

print(train_dataset)
class LyftModel(nn.Module):

    

    def __init__(self, cfg):

        super().__init__()

        

        # set pretrained=True while training

        self.backbone = resnet34(pretrained=False) 

        

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
# compiling model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LyftModel(training_cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
# get hardware type (CPU, GPU, TPU)

device
# training loop

tr_it = iter(train_dataloader)

progress_bar = tqdm(range(training_cfg["train_params"]["max_num_steps"]))



losses_train = []



for _ in progress_bar:

    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)

    model.train()

    torch.set_grad_enabled(True)

    

    # forward pass

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



    losses_train.append(loss.item())

        

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
# save full trained model

torch.save(model.state_dict(), f'model_state_last.pth')
# test configuration

test_cfg = inference_cfg["test_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(inference_cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(inference_cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])





print(test_dataloader)
# Saved state dict from the training notebook

WEIGHT_FILE = '/kaggle/input/lyft-l5-weights/model_state_last.pth'

model_state = torch.load(WEIGHT_FILE, map_location=device)

model.load_state_dict(model_state)
device
'''

model.eval()

torch.set_grad_enabled(False)



# store information for evaluation

future_coords_offsets_pd = []

timestamps = []



agent_ids = []

progress_bar = tqdm(test_dataloader)

for data in progress_bar:

    

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)



    outputs = model(inputs).reshape(targets.shape)

    

    future_coords_offsets_pd.append(outputs.cpu().numpy().copy())

    timestamps.append(data["timestamp"].numpy().copy())

    agent_ids.append(data["track_id"].numpy().copy())

'''
# submission.csv

'''

write_pred_csv('submission.csv',

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd),

              )

              

'''

model_sub = pd.read_csv('/kaggle/input/lyft-l5-inference-batch64-resnet18/submission.csv')

model_sub.to_csv('submission.csv', index = False)