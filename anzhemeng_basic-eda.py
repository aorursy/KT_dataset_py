# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.testing import assert_equal

import matplotlib.pyplot as plt

from tqdm import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer



from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points



from IPython.display import display, clear_output

from IPython.display import HTML

import PIL



import tifffile



from matplotlib import animation, rc



import seaborn as sns
multi_submission = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv')

single_submission = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv')
multi_submission.head()
single_submission.head()
assert_equal(multi_submission['timestamp'].unique(), single_submission['timestamp'].unique())

assert_equal(multi_submission['track_id'].unique(), single_submission['track_id'].unique())

assert_equal(multi_submission['conf_0'].unique(), single_submission['conf_0'].unique())
multi_submission.columns
single_submission.columns
assert_equal(multi_submission.columns.tolist(), single_submission.columns.tolist())
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

        'raster_size': [300, 300],

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

        'batch_size': 8,

        'shuffle': True,

        'num_workers': 4

    },

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 32,

        'shuffle': False,

        'num_workers': 4

    }

}



# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = '../input/lyft-motion-prediction-autonomous-vehicles'

dm = LocalDataManager(None)
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)
sample_zarr_dataset = ChunkedDataset('../input/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr')

sample_zarr_dataset.open()
print(sample_zarr_dataset)
frames = sample_zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = sample_zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]





plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, sample_zarr_dataset, rast)

data = dataset_ego[0]

im1 = data["image"].transpose(1, 2, 0)

im1 = dataset_ego.rasterizer.to_rgb(im1)



cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, sample_zarr_dataset, rast)

data = dataset_ego[0]

im2 = data["image"].transpose(1, 2, 0)

im2 = dataset_ego.rasterizer.to_rgb(im2)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im2, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)



_, ax = plt.subplots(1,2, figsize = (7, 7))

ax[0].imshow(im1[::-1])

ax[0].title.set_text('Object Detection')

ax[1].imshow(im2[::-1])

ax[1].title.set_text('Trajectory Simulation')

plt.show()
# keep animation in the notebook

def animate_solution(images):

    def animate(i):

        im.set_data(images[i])

        

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames = len(images), interval = 60)
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, sample_zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

HTML(animate_solution(images).to_jshtml())
test_zarr_dataset = ChunkedDataset('../input/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr')

test_zarr_dataset.open()
print(test_zarr_dataset)
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, test_zarr_dataset, rast)

data = dataset_ego[0]

im1 = data["image"].transpose(1, 2, 0)

im1 = dataset_ego.rasterizer.to_rgb(im1)



cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, test_zarr_dataset, rast)

data = dataset_ego[0]

im2 = data["image"].transpose(1, 2, 0)

im2 = dataset_ego.rasterizer.to_rgb(im2)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im2, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)



_, ax = plt.subplots(1,2, figsize = (7, 7))

ax[0].imshow(im1[::-1])

ax[0].title.set_text('Object Detection')

ax[1].imshow(im2[::-1])

ax[1].title.set_text('Trajectory Simulation')

plt.show()
frames = test_zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = test_zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]





plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, test_zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

HTML(animate_solution(images).to_jshtml())
train_zarr_dataset = ChunkedDataset('../input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr')

train_zarr_dataset.open()
print(train_zarr_dataset)
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, train_zarr_dataset, rast)

data = dataset_ego[0]

im1 = data["image"].transpose(1, 2, 0)

im1 = dataset_ego.rasterizer.to_rgb(im1)



cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, train_zarr_dataset, rast)

data = dataset_ego[0]

im2 = data["image"].transpose(1, 2, 0)

im2 = dataset_ego.rasterizer.to_rgb(im2)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im2, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)



_, ax = plt.subplots(1,2, figsize = (7, 7))

ax[0].imshow(im1[::-1])

ax[0].title.set_text('Object Detection')

ax[1].imshow(im2[::-1])

ax[1].title.set_text('Trajectory Simulation')

plt.show()
frames = train_zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = train_zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]





plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, train_zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

HTML(animate_solution(images).to_jshtml())
validate_zarr_dataset = ChunkedDataset('../input/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr')

validate_zarr_dataset.open()
print(validate_zarr_dataset)
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, validate_zarr_dataset, rast)

data = dataset_ego[0]

im1 = data["image"].transpose(1, 2, 0)

im1 = dataset_ego.rasterizer.to_rgb(im1)



cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset_ego = EgoDataset(cfg, validate_zarr_dataset, rast)

data = dataset_ego[0]

im2 = data["image"].transpose(1, 2, 0)

im2 = dataset_ego.rasterizer.to_rgb(im2)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im2, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)



_, ax = plt.subplots(1,2, figsize = (7, 7))

ax[0].imshow(im1[::-1])

ax[0].title.set_text('Object Detection')

ax[1].imshow(im2[::-1])

ax[1].title.set_text('Trajectory Simulation')

plt.show()
frames = validate_zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = validate_zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]





plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, validate_zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, yaws=data["target_yaws"], radius=1, rgb_color=TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

HTML(animate_solution(images).to_jshtml())
mask = np.load('../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz')

mask.f.arr_0
import zarr



z = zarr.open('../input/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr')

z.info
agents = z.agents.get_mask_selection(mask.f.arr_0)

agents
centroids = pd.DataFrame(agents['centroid'])

extents = pd.DataFrame(agents['extent'])

velocities = pd.DataFrame(agents['velocity'])

prob = pd.DataFrame(agents['label_probabilities'])
fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.scatter(centroids[0], centroids[1])

plt.xlabel('x', fontsize=11); plt.ylabel('y', fontsize=11)

plt.title("Centroids distribution (sample.zarr)")

plt.show()
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(2, 2, 1, projection='3d')

ax.scatter3D(extents[0], extents[1], extents[2], color='yellow')

ax.set_title('Scatter Distribution of Extents')



ax = fig.add_subplot(2, 2, 2)

sns.distplot(extents[0], color='red', ax=ax)

ax.set_title("Extent_0 Distribution")



ax = fig.add_subplot(2, 2, 3)

sns.distplot(extents[1], color='blue', ax=ax)

ax.set_title("Extent_1 Distribution")



ax = fig.add_subplot(2, 2, 4)

sns.distplot(extents[2], color='green', ax=ax)

ax.set_title("Extent_2 Distribution")



plt.tight_layout()
sns.jointplot(velocities[0], velocities[1]).plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.hist(prob.mean())