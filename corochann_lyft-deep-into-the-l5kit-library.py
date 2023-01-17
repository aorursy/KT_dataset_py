import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
import zarr



import l5kit

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



from matplotlib import animation, rc

from IPython.display import HTML



rc('animation', html='jshtml')

print("l5kit version:", l5kit.__version__)
from IPython.display import display, clear_output

import PIL





# Originally from https://www.kaggle.com/jpbremer/lyft-scene-visualisations by @jpbremer

# Modified following:

#  - Added to show timestamp

#  - Do not show image, to only show animation.

#  - Use blit=True.



def animate_solution(images, timestamps=None):

    def animate(i):

        changed_artifacts = [im]

        im.set_data(images[i])

        if timestamps is not None:

            time_text.set_text(timestamps[i])

            changed_artifacts.append(im)

        return tuple(changed_artifacts)



    

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    if timestamps is not None:

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)



    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=60, blit=True)

    

    # To prevent plotting image inline.

    plt.close()

    return anim
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")

print(cfg)
dm = LocalDataManager()

dataset_path = dm.require('scenes/sample.zarr')

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
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
def create_animate_for_indexes(dataset, indexes):

    images = []

    timestamps = []



    for idx in indexes:

        data = dataset[idx]

        im = data["image"].transpose(1, 2, 0)

        im = dataset.rasterizer.to_rgb(im)

        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

        clear_output(wait=True)

        images.append(PIL.Image.fromarray(im[::-1]))

        timestamps.append(data["timestamp"])



    anim = animate_solution(images, timestamps)

    return anim



def create_animate_for_scene(dataset, scene_idx):

    indexes = dataset.get_scene_indices(scene_idx)

    return create_animate_for_indexes(dataset, indexes)
dataset = dataset_dict["py_semantic"]

scene_idx = 34

anim = create_animate_for_scene(dataset, scene_idx)

print("scene_idx", scene_idx)

HTML(anim.to_jshtml())
scene_idx = 0

print("scene_idx", scene_idx)

anim = create_animate_for_scene(dataset, scene_idx)

display(HTML(anim.to_jshtml()))
scene_idx = 1

print("scene_idx", scene_idx)

anim = create_animate_for_scene(dataset, scene_idx)

display(HTML(anim.to_jshtml()))
scene_idx = 2

print("scene_idx", scene_idx)

anim = create_animate_for_scene(dataset, scene_idx)

display(HTML(anim.to_jshtml()))
semantic_rasterizer = rasterizer_dict["py_semantic"]

dataset = dataset_dict["py_semantic"]
# It shows the split point of each scene.

print("cumulative_sizes", dataset.cumulative_sizes)



# How's the length of each scene?

print("Each scene's length", dataset.cumulative_sizes[1:] - dataset.cumulative_sizes[:-1])
data = dataset[0]



print("dataset[0]=data is ", type(data))



def _describe(value):

    if hasattr(value, "shape"):

        return f"{type(value).__name__:20} shape={value.shape}"

    else:

        return f"{type(value).__name__:20} value={value}"



for key, value in data.items():

    print("  ", f"{key:25}", _describe(value))
scene_index = 0

frame_indices = dataset.get_scene_indices(scene_index)

print(f"frame_indices for scene {scene_index} = {frame_indices}")



scene_dataset = dataset.get_scene_dataset(scene_index)

print(f"scene_dataset {type(scene_dataset).__name__}, length {len(scene_dataset)}")



# Animate whole "scene_dataset"

create_animate_for_indexes(scene_dataset, np.arange(len(scene_dataset)))
frame_idx = 10

indices = dataset.get_frame_indices(frame_idx)



# These are same for EgoDataset!

print(f"frame_idx = {frame_idx}, indices = {indices}")
semantic_rasterizer = rasterizer_dict["py_semantic"]

agent_dataset = AgentDataset(cfg, zarr_dataset, semantic_rasterizer)



print(f"EgoDataset size {len(dataset)}, AgentDataset size {len(agent_dataset)}")
# The returned data structure is same.

data = agent_dataset[0]



print("agent_dataset[0]=data is ", type(data))



def _describe(value):

    if hasattr(value, "shape"):

        return f"{type(value).__name__:20} shape={value.shape}"

    else:

        return f"{type(value).__name__:20} value={value}"



for key, value in data.items():

    print("  ", f"{key:25}", _describe(value))
scene_index = 3

frame_indices = agent_dataset.get_scene_indices(scene_index)

print(f"frame_indices for scene {scene_index} = {frame_indices}")



scene_dataset = agent_dataset.get_scene_dataset(scene_index)

print(f"scene_dataset {type(scene_dataset).__name__}, length {len(scene_dataset)}")



# Animate whole "scene_dataset"

create_animate_for_indexes(scene_dataset, np.arange(len(scene_dataset)))
for i in range(1000):

    print(i, agent_dataset.get_frame_indices(i))
frame_indices = agent_dataset.get_frame_indices(648)



fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 5))

axes = axes.flatten()



for i in range(len(frame_indices)):

    index = frame_indices[i]

    t = agent_dataset[index]["timestamp"]

    # Timestamp is same for same frame.

    print(f"timestamp = {t}")

    visualize_rgb_image(agent_dataset, index=index, title=f"index={index}", ax=axes[i])

fig.show()
print("scenes", zarr_dataset.scenes)

print("scenes[0]", zarr_dataset.scenes[0])
print("frames", zarr_dataset.frames)

print("frames[0]", zarr_dataset.frames[0])
print("agents", zarr_dataset.agents)

print("agents[0]", zarr_dataset.agents[0])
print("tl_faces", zarr_dataset.tl_faces)

print("tl_faces[0]", zarr_dataset.tl_faces[0])