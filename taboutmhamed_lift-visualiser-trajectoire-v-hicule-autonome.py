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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pymap3d==2.1.0

!pip install l5kit

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

# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("/kaggle/input/lyftconfigfiles/visualisation_config.yaml")

print(cfg)
print(f'current raster_param:\n')

for k,v in cfg["raster_params"].items():

    print(f"{k}:{v}")
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
frames = zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]





plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])
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
rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)
data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[0]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

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