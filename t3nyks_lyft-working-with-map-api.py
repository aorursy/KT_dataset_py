!pip install --upgrade pip
!pip install pymap3d==2.1.0
!pip install -U l5kit
import numpy as np
import l5kit, os
import matplotlib.pyplot as plt
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
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
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
def is_junction(elem, map_api):
    return elem.element.HasField("junction")

def get_junctions(map_api):
    return [elem for elem in map_api.elements if is_junction(elem, map_api)]

all_junctions = get_junctions(map_api)
all_junctions[0]
def is_node(elem, map_api):
    return elem.element.HasField("node")

def get_nodes(map_api):
    return [elem for elem in map_api.elements if is_junction(elem, map_api)]

all_nodes = get_nodes(map_api)
all_nodes[0]
def is_segment(elem, map_api):
    return elem.element.HasField("segment")

def get_segments(map_api):
    return [elem for elem in map_api.elements if is_segment(elem, map_api)]

all_segments = get_segments(map_api)
all_segments[0]
def is_lane(elem, map_api):
    return elem.element.HasField("lane")

def get_lanes(map_api):
    return [elem for elem in map_api.elements if is_lane(elem, map_api)]

all_lanes = get_lanes(map_api)
all_lanes[0]
def is_traffic_element(elem, map_api):
    return elem.element.HasField("traffic_control_element")

def get_traffic_elements(map_api):
    return [elem for elem in map_api.elements if is_traffic_element(elem, map_api)]

all_traffic_elements = get_traffic_elements(map_api)
all_traffic_elements[0]
def is_segment_sequence(elem, map_api):
    return elem.element.HasField("segment_sequence")

def get_segment_sequences(map_api):
    return [elem for elem in map_api.elements if is_segment_sequence(elem, map_api)]

all_segment_sequences = get_segment_sequences(map_api)
print(len(all_segment_sequences))
def is_annotated_shape(elem, map_api):
    return elem.element.HasField("annotated_shape")

def get_annotated_shapes(map_api):
    return [elem for elem in map_api.elements if is_annotated_shape(elem, map_api)]

all_annotated_shapes = get_annotated_shapes(map_api)
all_annotated_shapes[0]
