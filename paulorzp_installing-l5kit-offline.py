# Installing l5kit offline

!pip install --no-index -f ../input/downloading-l5kit pip==20.2.2 >/dev/nul

!pip install --no-index -f ../input/downloading-l5kit -U l5kit > /dev/nul
import numpy as np

import pandas as pd

import os



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



import zarr



import matplotlib.pyplot as plt

from matplotlib import animation, rc

from IPython.display import HTML, display, clear_output

import PIL

rc('animation', html='jshtml')
# set env variable for data

DIR_PATH = "../input/lyft-motion-prediction-autonomous-vehicles/"

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open();
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[80]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = plt.subplots(figsize=(10,10))

plt.imshow(im[::-1])

plt.show()
dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
#sample submission

sample_submission = pd.read_csv(f'{DIR_PATH}multi_mode_sample_submission.csv')

sample_submission.to_csv("submission.csv",index=False)