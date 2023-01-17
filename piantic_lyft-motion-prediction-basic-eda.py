from IPython.display import HTML

HTML('<center><iframe width="640" height="360" src="https://player.vimeo.com/video/389096888" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import os

from os import listdir

import pandas as pd



import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import numpy as np



#color

from colorama import Fore, Back, Style



# Suppress warnings

import warnings

warnings.filterwarnings('ignore')
# List files available

list(os.listdir("../input/lyft-motion-prediction-autonomous-vehicles"))
single_mode_sample_submission = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv')

multi_mode_sample_submission = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv')
print(Fore.YELLOW + 'Sample submission for single mode shape: ',Style.RESET_ALL,single_mode_sample_submission.shape)

single_mode_sample_submission.head(5)
print(Fore.BLUE + 'Sample submission for multi mode shape: ',Style.RESET_ALL,multi_mode_sample_submission.shape)

multi_mode_sample_submission.head(5)
# Null values and Data types

print(Fore.YELLOW + 'Single Mode Sample Submission !!',Style.RESET_ALL)

print(single_mode_sample_submission.info())

print('-------------')

print(Fore.BLUE + 'Multi Mode Sample Submission !!',Style.RESET_ALL)

print(multi_mode_sample_submission.info())
!pip install --upgrade pip > /dev/null 

!pip uninstall typing -y > /dev/null 

!pip install --ignore-installed --target=/kaggle/working l5kit > /dev/null 
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
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
cfg = {

    #'format_version': 4,   

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

    

    'val_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 16

    },

}
print(f'current raster_param:\n')

for k,v in cfg["raster_params"].items():

    print(Fore.YELLOW + f"{k}",Style.RESET_ALL + f":{v}")
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
import zarr

train_zarr = zarr.open("../input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr")



print(type(train_zarr))
train_zarr.info
fields = [

    "Num Scenes",

    "Num Frames",

    "Num Agents",

    "Total Time (hr)",

    "Avg Frames per Scene",

    "Avg Agents per Frame",

    "Avg Scene Time (sec)",

    "Avg Frame frequency",

]
if len(zarr_dataset.frames) > 1:

    times = zarr_dataset.frames[1:50]["timestamp"] - zarr_dataset.frames[0:49]["timestamp"]

    frequency = np.mean(1 / (times / 1e9))  # from nano to sec

else:

    print(f"warning, not enough frames({len(zarr_dataset.frames)}) to read the frequency, 10 will be set")

    frequency = 10
values = [

    len(zarr_dataset.scenes),

    len(zarr_dataset.frames),

    len(zarr_dataset.agents),

    len(zarr_dataset.frames) / max(frequency, 1) / 3600,

    len(zarr_dataset.frames) / max(len(zarr_dataset.scenes), 1),

    len(zarr_dataset.agents) / max(len(zarr_dataset.frames), 1),

    len(zarr_dataset.frames) / max(len(zarr_dataset.scenes), 1) / frequency,

    frequency,

]
table = PrettyTable(field_names=[fields[0]])

table.add_row([values[0]])
print(Fore.YELLOW + str(table) + Style.RESET_ALL)
print(Fore.YELLOW + table.get_string(fields=["Num Scenes"]) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[1]])

table.add_row([values[1]])

print(Fore.BLUE + str(table) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[2]])

table.add_row([values[2]])

print(Fore.YELLOW + str(table) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[3]])

table.float_format = ".2"

table.add_row([values[3]])

print(Fore.BLUE + str(table) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[4]])

table.float_format = ".2"

table.add_row([values[4]])

print(Fore.YELLOW + str(table) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[5]])

table.float_format = ".2"

table.add_row([values[5]])

print(Fore.BLUE + str(table) + Style.RESET_ALL)
table = PrettyTable(field_names=[fields[6]])

table.float_format = ".2"

table.add_row([values[6]])

print(Fore.YELLOW + str(table) + Style.RESET_ALL)
agents = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles-as-csv/agents_0_10019001_10019001.csv')

agents
cont_feats = ["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw"]

fig = px.imshow(agents[cont_feats].corr(),

                labels=dict(x="Correlation of features", y="", color="Correlation"),

                x=["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw"],

                y=["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw"]

               )

plt.figure(figsize=(16,12));

fig.update_xaxes(side="top")

fig.show()
fig = plt.figure(figsize=(16, 12));

sns.distplot(agents['centroid_x'], color='steelblue');

sns.distplot(agents['centroid_y'], color='red');

plt.title("Distributions of Centroid X and Y");
fig = plt.figure(figsize=(16, 12));

sns.distplot(agents['extent_x'], color='steelblue');

sns.distplot(agents['extent_y'], color='red');



plt.title("Distributions of Extents X and Y");
fig = plt.figure(figsize=(16, 12));

sns.distplot(agents['extent_z'], color='blue');



plt.title("Distributions of Extents z");
fig = plt.figure(figsize=(16, 12));

sns.distplot(agents['yaw'], color='blue');



plt.title("Distributions of Extents z");
frms = pd.read_csv("../input/lyft-motion-prediction-autonomous-vehicles-as-csv/frames_0_124167_124167.csv")

frms.head()
import seaborn as sns

colormap = plt.cm.magma

cont_feats = ["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"]

plt.figure(figsize=(16,12));

plt.title('Pearson correlation of features', y=1.05, size=15);

sns.heatmap(frms[cont_feats].corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True)
cont_feats = ["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"]

fig = px.imshow(frms[cont_feats].corr(),

                labels=dict(x="Correlation of features", y="", color="Correlation"),

                x=["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"],

                y=["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"]

               )

plt.figure(figsize=(16,12));

fig.update_xaxes(side="top")

fig.show()
import numpy as np

zero_count_list, one_count_list = [], []

cols_list = ["label_probabilities_PERCEPTION_LABEL_UNKNOWN","label_probabilities_PERCEPTION_LABEL_CAR","label_probabilities_PERCEPTION_LABEL_CYCLIST","label_probabilities_PERCEPTION_LABEL_PEDESTRIAN"]

for col in cols_list:

    zero_count_list.append((agents[col]==0).sum())

    one_count_list.append((agents[col]==1).sum())



N = len(cols_list)

ind = np.arange(N)

width = 0.35



plt.figure(figsize=(6,10))

p1 = plt.barh(ind, zero_count_list, width, color='red')

p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="blue")

plt.yticks(ind, cols_list)

plt.legend((p1[0], p2[0]), ('Zero count', 'One Count'))

plt.show()
#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
cfg = {

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

    

    'val_data_loader': {

        'key': 'scenes/sample.zarr',

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 16

    },

}
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
agents = zarr_dataset.agents

agents_df = pd.DataFrame(agents)

agents_df.columns = ["data"]; features = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities']



for i, feature in enumerate(features):

    agents_df[feature] = agents_df['data'].apply(lambda x: x[i])

agents_df.drop(columns=["data"],inplace=True)

print(f"agents dataset: {agents_df.shape}")

agents_df.head()
agents_df['cx'] = agents_df['centroid'].apply(lambda x: x[0])

agents_df['cy'] = agents_df['centroid'].apply(lambda x: x[1])



fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.scatter(agents_df['cx'], agents_df['cy'], marker='+')

plt.xlabel('x', fontsize=11); plt.ylabel('y', fontsize=11)

plt.title("Centroids distribution")

plt.show()
agents_df['ex'] = agents_df['extent'].apply(lambda x: x[0])

agents_df['ey'] = agents_df['extent'].apply(lambda x: x[1])

agents_df['ez'] = agents_df['extent'].apply(lambda x: x[2])



sns.set_style('whitegrid')



fig, ax = plt.subplots(1,3,figsize=(16,5))

plt.subplot(1,3,1)

plt.scatter(agents_df['ex'], agents_df['ey'], marker='+')

plt.xlabel('ex', fontsize=11); plt.ylabel('ey', fontsize=11)

plt.title("Extent: ex-ey")

plt.subplot(1,3,2)

plt.scatter(agents_df['ey'], agents_df['ez'], marker='+', color="red")

plt.xlabel('ey', fontsize=11); plt.ylabel('ez', fontsize=11)

plt.title("Extent: ey-ez")

plt.subplot(1,3,3)

plt.scatter(agents_df['ez'], agents_df['ex'], marker='+', color="green")

plt.xlabel('ez', fontsize=11); plt.ylabel('ex', fontsize=11)

plt.title("Extent: ez-ex")

plt.show();
agents_df['vx'] = agents_df['velocity'].apply(lambda x: x[0])

agents_df['vy'] = agents_df['velocity'].apply(lambda x: x[1])



fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.title("Velocity distribution")

plt.scatter(agents_df['vx'], agents_df['vy'], marker='.', color="red")

plt.xlabel('vx', fontsize=11); plt.ylabel('vy', fontsize=11)

plt.show();
scenes = zarr_dataset.scenes

scenes_df = pd.DataFrame(scenes)

scenes_df.columns = ["data"]; features = ['frame_index_interval', 'host', 'start_time', 'end_time']

for i, feature in enumerate(features):

    scenes_df[feature] = scenes_df['data'].apply(lambda x: x[i])

scenes_df.drop(columns=["data"],inplace=True)

print(f"scenes dataset: {scenes_df.shape}")

scenes_df.head()
f, ax = plt.subplots(1,1, figsize=(6,4))

sns.countplot(scenes_df.host)

plt.xlabel('Host')

plt.ylabel(f'Count')

plt.title("Scenes host count distribution")

plt.show()
scenes_df['frame_index_start'] = scenes_df['frame_index_interval'].apply(lambda x: x[0])

scenes_df['frame_index_end'] = scenes_df['frame_index_interval'].apply(lambda x: x[1])

scenes_df.head()
frames = zarr_dataset.frames

coords = np.zeros((len(frames), 2))

for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    frame = zarr_dataset.frames[idx_data]

    coords[idx_coord] = frame["ego_translation"][:2]
fig = go.Figure(data=go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='lines'))

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='lines', name='lines'))

fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='markers', name='lines+markers'))

fig.show()
frames_df = pd.DataFrame(zarr_dataset.frames)

frames_df.columns = ["data"]; features = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 

                                          'ego_translation','ego_rotation']

for i, feature in enumerate(features):

    frames_df[feature] = frames_df['data'].apply(lambda x: x[i])

frames_df.drop(columns=["data"],inplace=True)

print(f"frames dataset: {frames_df.shape}")

frames_df.head()
frames_df['dx'] = frames_df['ego_translation'].apply(lambda x: x[0])

frames_df['dy'] = frames_df['ego_translation'].apply(lambda x: x[1])

frames_df['dz'] = frames_df['ego_translation'].apply(lambda x: x[2])
sns.set_style('whitegrid')

plt.figure()



fig, ax = plt.subplots(1,3,figsize=(16,5))



plt.subplot(1,3,1)

plt.scatter(frames_df['dx'], frames_df['dy'], marker='+')

plt.xlabel('dx', fontsize=11); plt.ylabel('dy', fontsize=11)

plt.title("Translations: dx-dy")

plt.subplot(1,3,2)

plt.scatter(frames_df['dy'], frames_df['dz'], marker='+', color="red")

plt.xlabel('dy', fontsize=11); plt.ylabel('dz', fontsize=11)

plt.title("Translations: dy-dz")

plt.subplot(1,3,3)

plt.scatter(frames_df['dz'], frames_df['dx'], marker='+', color="green")

plt.xlabel('dz', fontsize=11); plt.ylabel('dx', fontsize=11)

plt.title("Translations: dz-dx")



fig.suptitle("Ego translations in 2D planes of the 3 components (dx,dy,dz)")

plt.show();
fig, ax = plt.subplots(3,3,figsize=(12,12))

colors = ['magenta', 'orange', 'darkblue', 'black', 'cyan', 'darkgreen', 'red', 'blue', 'green']

for i in range(0,3):

    for j in range(0,3):

        df = frames_df['ego_rotation'].apply(lambda x: x[i][j])

        plt.subplot(3,3,i * 3 + j + 1)

        sns.distplot(df, hist=False, color = colors[ i * 3 + j  ])

        plt.xlabel(f'r[ {i + 1} ][ {j + 1} ]')

fig.suptitle("Ego rotation angles distribution")

plt.show()
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
cfg = {

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

    'val_data_loader': {

        'key': 'scenes/sample.zarr',

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 16

    },

}
rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)
from l5kit.geometry import transform_points



from l5kit.visualization import (draw_trajectory,       # draws 2D trajectories from coordinates and yaws offset on an image

                                 TARGET_POINTS_COLOR)



data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = px.imshow(im[::-1])

fig.show()
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)



# EgoDataset object

dataset = EgoDataset(cfg, zarr_dataset, rast)

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = px.imshow(im[::-1], title='Satellite View: Ground Truth Trajectory of Autonomous Vehicle')

fig.show()
dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[50]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = px.imshow(im[::-1])

fig.update_layout(

    title={

        'text': "Agent",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
from IPython.display import display, clear_output

from IPython.display import HTML



import PIL

from matplotlib import animation
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



fig = plt.figure(figsize = (10,10))



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    im = plt.imshow(PIL.Image.fromarray(im[::-1]), animated=True)

    plt.axis("off")

    images.append([im])

ani = animation.ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
HTML(ani.to_jshtml())
cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



fig = plt.figure(figsize = (10,10))



for idx in indexes:

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    im = plt.imshow(PIL.Image.fromarray(im[::-1]), animated=True)

    plt.axis("off")

    images.append([im])

ani = animation.ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
HTML(ani.to_jshtml())
# satellite view

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



fig = plt.figure(figsize = (10,10))



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    im = plt.imshow(PIL.Image.fromarray(im[::-1]), animated=True)

    plt.axis("off")

    images.append([im])

ani = animation.ArtistAnimation(fig, images, interval=100, blit=False, repeat_delay=1000)
HTML(ani.to_jshtml())
from l5kit.data.map_api import MapAPI

from l5kit.rasterization.rasterizer_builder import _load_metadata



semantic_map_filepath = dm.require(cfg["raster_params"]["semantic_map_key"])

dataset_meta = _load_metadata(cfg["raster_params"]["dataset_meta_key"], dm)

world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)



map_api = MapAPI(semantic_map_filepath, world_to_ecef)

MAP_LAYERS = ["junction", "node", "segment", "lane"]





def element_of_type(elem, layer_name):

    return elem.element.HasField(layer_name)





def get_elements_from_layer(map_api, layer_name):

    return [elem for elem in map_api.elements if element_of_type(elem, layer_name)]





class MapRenderer:

    

    def __init__(self, map_api):

        self._color_map = dict(drivable_area='#a6cee3',

                               road_segment='#1f78b4',

                               road_block='#b2df8a',

                               lane='#474747')

        self._map_api = map_api

    

    def render_layer(self, layer_name):

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        

    def render_lanes(self):

        all_lanes = get_elements_from_layer(self._map_api, "lane")

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        for lane in all_lanes:

            self.render_lane(ax, lane)

        return fig, ax

        

    def render_lane(self, ax, lane):

        coords = self._map_api.get_lane_coords(MapAPI.id_as_str(lane.id))

        self.render_boundary(ax, coords["xyz_left"])

        self.render_boundary(ax, coords["xyz_right"])

        

    def render_boundary(self, ax, boundary):

        xs = boundary[:, 0]

        ys = boundary[:, 1] 

        ax.plot(xs, ys, color=self._color_map["lane"], label="lane")

        

        

renderer = MapRenderer(map_api)

fig, ax = renderer.render_lanes()
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