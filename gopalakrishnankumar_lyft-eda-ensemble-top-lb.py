from IPython.display import HTML



HTML('<center><iframe  width="850" height="450" src="https://www.youtube.com/embed/K0H43N-Hx7w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
!pip install pymap3d==2.1.0

!pip install -U l5kit
import os, gc

import zarr

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

sns.set()

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



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
def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")

print(cfg)
dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
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
data = dataset[80]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



plt.imshow(im[::-1])

plt.show()
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

data = dataset[80]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = plt.subplots(figsize=(10,10))

plt.imshow(im[::-1])

plt.show()
dataset = AgentDataset(cfg, zarr_dataset, rast)

data = dataset[80]



im = data["image"].transpose(1, 2, 0)

im = dataset.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)



fig = plt.subplots(figsize=(10,10))

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
anim = animate_solution(images)

HTML(anim.to_jshtml())
anim = animate_solution(images)

HTML(anim.to_jshtml())
z = zarr.open("./dataset.zarr", mode="w", shape=(500,), dtype=np.float32, chunks=(100,))



# We can write to it by assigning to it. This gets persisted on disk.

z[0:150] = np.arange(150)

print(z.info)



# Reading from a zarr array is as easy as slicing from it like you would any numpy array. 

# The return value is an ordinary numpy array. Zarr takes care of determining which chunks to read from.

print(z[:10])

print(z[::20]) # Read every 20th value
SCENE_DTYPE = [

    ("frame_index_interval", np.int64, (2,)),

    ("host", "<U16"),  # Unicode string up to 16 chars

    ("start_time", np.int64),

    ("end_time", np.int64),

]
FRAME_DTYPE = [

    ("timestamp", np.int64),

    ("agent_index_interval", np.int64, (2,)),

    ("traffic_light_faces_index_interval", np.int64, (2,)),

    ("ego_translation", np.float64, (3,)),

    ("ego_rotation", np.float64, (3, 3)),

]
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/tlThdr3O5Qo?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

dm = LocalDataManager()

sample_path = '../input/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr'

sample_dataset = ChunkedDataset(sample_path)

sample_dataset.open()



sample_agents = sample_dataset.agents

sample_agents = pd.DataFrame(sample_agents)

sample_agents.columns = ["data"]; features = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities']



for i, feature in enumerate(features):

    sample_agents[feature] = sample_agents['data'].apply(lambda x: x[i])

sample_agents.drop(columns=["data"],inplace=True)

sample_agents.head()
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
multisub = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv')
cols = list(multisub.columns)

confs = cols[2:5]

conf0 = cols[5:105]

conf1 = cols[105:205]

conf2 = cols[205:305]
!ls ../input
sub0 = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv')

sub1 = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv')

sub2 = pd.read_csv('../input/lyft-constant-velocity-extrapolation-baseline/submission.csv')
multisub[confs] = [.49,.2,.31]

multisub[conf0] = sub0[conf0]

multisub[conf1] = sub1[conf0]

multisub[conf2] = sub2[conf0]
multisub
multisub.to_csv('submission.csv', index=False, float_format='%.6g')