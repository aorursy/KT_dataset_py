from IPython.display import Image

!ls ../input/imagelyft
Image("../input/imagelyft/pipeline.png")
Image("../input/imagelyft/av.jpg")
!pip install --upgrade pip -q

!pip install pymap3d==2.1.0 -q

!pip install -U l5kit -q
import l5kit, os

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")
import matplotlib.pyplot as plt



import numpy as np



from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset



from matplotlib import animation, rc

from IPython.display import HTML



rc('animation', html='jshtml')
from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
print(f'current raster_param:\n')

for k,v in cfg["raster_params"].items():

    print(f"{k}:{v}")


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
import numpy as np

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
data = dataset[80]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
import numpy as np

from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_satellite"

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

anim = animate_solution(images)

HTML(anim.to_jshtml())
dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[0]



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