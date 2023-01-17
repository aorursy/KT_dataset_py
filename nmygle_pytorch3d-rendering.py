!pip install pymap3d==2.1.0
!pip install protobuf==3.12.2
!pip install transforms3d
!pip install zarr
!pip install ptable
!pip install --no-dependencies l5kit
!conda install pytorch3d -c pytorch3d -y
%%writefile tripy.py
# MIT License
#
# Copyright (c) 2017 Sam Bolgert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# https://github.com/linuxlewis/tripy
#
import math
import sys
from collections import namedtuple

Point = namedtuple('Point', ['idx', 'x', 'y', 'z']) # append point id

EPSILON = math.sqrt(sys.float_info.epsilon)


def earclip(polygon):
    """
    Simple earclipping algorithm for a given polygon p.
    polygon is expected to be an array of 2-tuples of the cartesian points of the polygon
    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as an array of 3-tuples where each item in the tuple is a 2-tuple of the cartesian point.
    e.g
    > polygon = [(0,1), (-1, 0), (0, -1), (1, 0)]
    > triangles = tripy.earclip(polygon)
    > triangles
    [((1, 0), (0, 1), (-1, 0)), ((1, 0), (-1, 0), (0, -1))]
    Implementation Reference:
        - https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    """
    ear_vertex = []
    triangles = []

    polygon = [Point(idx, *point) for idx, point in enumerate(polygon)]

    if _is_clockwise(polygon):
        polygon.reverse()

    point_count = len(polygon)
    for i in range(point_count):
        prev_index = i - 1
        prev_point = polygon[prev_index]
        point = polygon[i]
        next_index = (i + 1) % point_count
        next_point = polygon[next_index]

        if _is_ear(prev_point, point, next_point, polygon):
            ear_vertex.append(point)

    while ear_vertex and point_count >= 3:
        ear = ear_vertex.pop(0)
        i = polygon.index(ear)
        prev_index = i - 1
        prev_point = polygon[prev_index]
        next_index = (i + 1) % point_count
        next_point = polygon[next_index]

        polygon.remove(ear)
        point_count -= 1
        #triangles.append(((prev_point.x, prev_point.y, prev_point.z),
        #                  (ear.x, ear.y, ear.z),
        #                  (next_point.x, next_point.y, next_point.z)))
        triangles.append((prev_point.idx, ear.idx, next_point.idx)) # append point id

        if point_count > 3:
            prev_prev_point = polygon[prev_index - 1]
            next_next_index = (i + 1) % point_count
            next_next_point = polygon[next_next_index]

            groups = [
                (prev_prev_point, prev_point, next_point, polygon),
                (prev_point, next_point, next_next_point, polygon),
            ]
            for group in groups:
                p = group[1]
                if _is_ear(*group):
                    if p not in ear_vertex:
                        ear_vertex.append(p)
                elif p in ear_vertex:
                    ear_vertex.remove(p)
    return triangles


def _is_clockwise(polygon):
    s = 0
    polygon_count = len(polygon)
    for i in range(polygon_count):
        point = polygon[i]
        point2 = polygon[(i + 1) % polygon_count]
        s += (point2.x - point.x) * (point2.y + point.y)
    return s > 0


def _is_convex(prev, point, next_point):
    return _triangle_sum(prev.x, prev.y, point.x, point.y, next_point.x, next_point.y) < 0


def _is_ear(p1, p2, p3, polygon):
    ear = _contains_no_points(p1, p2, p3, polygon) and \
          _is_convex(p1, p2, p3) and \
          _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
    return ear


def _contains_no_points(p1, p2, p3, polygon):
    for pn in polygon:
        if pn in (p1, p2, p3):
            continue
        elif _is_point_inside(pn, p1, p2, p3):
            return False
    return True


def _is_point_inside(p, a, b, c):
    area = _triangle_area(a.x, a.y, b.x, b.y, c.x, c.y)
    area1 = _triangle_area(p.x, p.y, b.x, b.y, c.x, c.y)
    area2 = _triangle_area(p.x, p.y, a.x, a.y, c.x, c.y)
    area3 = _triangle_area(p.x, p.y, a.x, a.y, b.x, b.y)
    areadiff = abs(area - sum([area1, area2, area3])) < EPSILON
    return areadiff


def _triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def _triangle_sum(x1, y1, x2, y2, x3, y3):
    return x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)


def calculate_total_area(triangles):
    result = []
    for triangle in triangles:
        sides = []
        for i in range(3):
            next_index = (i + 1) % 3
            pt = triangle[i]
            pt2 = triangle[next_index]
            # Distance between two points
            side = math.sqrt(math.pow(pt2[0] - pt[0], 2) + math.pow(pt2[1] - pt[1], 2))
            sides.append(side)
        # Heron's numerically stable forumla for area of a triangle:
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        # However, for line-like triangles of zero area this formula can produce an infinitesimally negative value
        # as an input to sqrt() due to the cumulative arithmetic errors inherent to floating point calculations:
        # https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
        # For this purpose, abs() is used as a reasonable guard against this condition.
        c, b, a = sorted(sides)
        area = .25 * math.sqrt(abs((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))))
        result.append((area, a, b, c))
    triangle_area = sum(tri[0] for tri in result)
    return triangle_area
%%writefile create_map_mesh.py
import pickle
import os
from tqdm import tqdm
import numpy as np

from l5kit.data.map_api import MapAPI
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization.rasterizer_builder import (_load_metadata, get_hardcoded_world_to_ecef)
import transforms3d

from tripy import earclip

def create_lane_surface(map_api):
    verts = []
    faces = []
    count = 0
    for element in tqdm(map_api):
        element_id = MapAPI.id_as_str(element.id)
        if map_api.is_lane(element):
            lane = map_api.get_lane_coords(element_id)
            left = lane["xyz_left"]
            right = lane["xyz_right"]
            points = np.vstack((left, np.flip(right, 0)))
            triangles = np.array(earclip(points)) + count
            verts.append(points)
            faces.append(triangles)
            count += points.shape[0]
    return verts, faces

def create_crosswalks(map_api):
    verts = []
    faces = []
    count = 0
    for element in tqdm(map_api):
        element_id = MapAPI.id_as_str(element.id)
        if map_api.is_crosswalk(element):
            crosswalk = map_api.get_crosswalk_coords(element_id)
            points = crosswalk["xyz"]
            triangles = np.array(earclip(points)) + count
            verts.append(points)
            faces.append(triangles)
            count += points.shape[0]
    return verts, faces

def create_map_mesh():
    filename = "mesh.p"

    os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
    dm = LocalDataManager(None)

    # Rasterizer
    semantic_map_path = dm.require("semantic_map/semantic_map.pb")
    dataset_meta = _load_metadata("meta.json", dm)
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    map_api = MapAPI(semantic_map_path, world_to_ecef)
    lanes_verts, lanes_faces = create_lane_surface(map_api)
    cw_verts, cw_faces = create_crosswalks(map_api)
    with open(filename, "wb") as fp:
        data = {
            "lane": [lanes_verts, lanes_faces],
            "cw": [cw_verts, cw_faces]
        }
        pickle.dump(data, fp)

if __name__ == "__main__":
    create_map_mesh()
# preprocess: create map mesh data
# !python create_map_mesh.py
!ls /kaggle/input/lyft-map-mesh
%%writefile pytorch3d_rasterizer.py
from typing import List, Optional, Tuple, Union
import pickle
import numpy as np

from l5kit.data.filter import (filter_agents_by_labels, filter_agents_by_track_id)
from l5kit.data.map_api import MapAPI
from l5kit.geometry import world_to_image_pixels_matrix, transform_point, transform_points, yaw_as_rotation33
from l5kit.rasterization import Rasterizer
from l5kit.rasterization.box_rasterizer import get_ego_as_agent

import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer import (
    FoVPerspectiveCameras, FoVOrthographicCameras,
    look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    SoftPhongShader, HardFlatShader
)

import transforms3d

from tripy import earclip


def create_agetns(agents):
    verts = []
    faces = []
    count = 0
    corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    for idx, agent in enumerate(agents):
        corners = corners_base_coords * agent["extent"][:2] / 2
        r_m = yaw_as_rotation33(agent["yaw"])
        points = transform_points(corners, r_m) + agent["centroid"][:2]
        points = np.concatenate([points, np.ones([points.shape[0], 1]) * 0.03], axis=1)
        triangles = np.array(earclip(points)) + count
        verts.append(points)
        faces.append(triangles)
        count += points.shape[0]
    return verts, faces

class PyTorch3dSemanticRasterizer(Rasterizer):

    def __init__(self,
                 raster_size: Tuple[int, int],
                 pixel_size: Union[np.ndarray, list, float],
                 ego_center: np.ndarray,
                 filter_agents_threshold: float,
                 history_num_frames: int,
                 semantic_map_path: str,
                 world_to_ecef: np.ndarray,
                 filename: str):
        super().__init__()

        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center

        #if isinstance(pixel_size, np.ndarray) or isinstance(pixel_size, list):
        #    self.pixel_size = pixel_size[0]

        self.filter_agents_threshold = filter_agents_threshold

        self.proto_API = MapAPI(semantic_map_path, world_to_ecef)

        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            lanes_verts, lanes_faces = data["lane"]
            cw_verts, cw_faces = data["cw"]
        
        self.device = torch.device("cuda:0")

        # lanes mesh
        lanes_verts_t = torch.Tensor(np.concatenate(lanes_verts, axis=0))
        lanes_verts_t[:,2] = 1 # z axis
        lanes_faces_t = torch.Tensor(np.concatenate(lanes_faces, axis=0))
        lanes_verts_rgb = torch.zeros_like(lanes_verts_t)[None]  # (1, V, 3)
        lanes_textures = TexturesVertex(verts_features=lanes_verts_rgb)
        lanes_mesh = Meshes(verts=[lanes_verts_t], faces=[lanes_faces_t], textures=lanes_textures).to(self.device)

        # crosswalks mesh
        cw_verts_t = torch.Tensor(np.concatenate(cw_verts, axis=0))
        cw_verts_t[:,2] = 2 # z axis
        cw_faces_t = torch.Tensor(np.concatenate(cw_faces, axis=0))
        cw_verts_rgb = torch.zeros_like(cw_verts_t)[None]  # (1, V, 3)
        cw_verts_rgb[0,:] = torch.Tensor(np.array([255 / 255, 255 / 255, 0 / 255], dtype=np.float32))
        cw_textures = TexturesVertex(verts_features=cw_verts_rgb)
        cw_mesh = Meshes(verts=[cw_verts_t], faces=[cw_faces_t], textures=cw_textures).to(self.device)

        # join mesh
        self.map_mesh = join_meshes_as_scene([lanes_mesh, cw_mesh])

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0, 
            faces_per_pixel=1,
        )

        cameras = FoVPerspectiveCameras(device=self.device)

        # lights = PointLights(device=device, location=((0.0, 0.0, -200.0),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            # shader=SoftSilhouetteShader(device=device, cameras=cameras)
            # shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
            shader=HardFlatShader(device=self.device, cameras=cameras)
        )
  
    def rasterize(self, history_frames: np.ndarray, history_agents: List[np.ndarray],
                  history_tl_faces: List[np.ndarray], agent: Optional[np.ndarray] = None,
                  agents: Optional[np.ndarray] = None
                  ) -> torch.Tensor:

        with torch.no_grad():
            frame = history_frames[0]
            if agent is None:
                translation = frame["ego_translation"]
                yaw = frame["ego_rotation"]
            else:
                translation = agent["centroid"]
                yaw = agent["yaw"]

            world_to_image_space = world_to_image_pixels_matrix(
                self.raster_size, self.pixel_size, translation, yaw, self.ego_center,
            )
            
            center_pixel = np.asarray(self.raster_size) * (0.5, 0.5)
            center_world = transform_point(center_pixel, np.linalg.inv(world_to_image_space))

            for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
                agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
                av_agent = get_ego_as_agent(frame).astype(agents.dtype)
            
                agents_verts, agents_faces = create_agetns(np.append(agents, av_agent))
                agents_verts_t = torch.Tensor(np.concatenate(agents_verts, axis=0)).to(self.device, non_blocking=True)
                agents_verts_t[:,2] = 3
                agents_faces_t = torch.Tensor(np.concatenate(agents_faces, axis=0)).to(self.device, non_blocking=True)
                agents_verts_rgb = torch.ones_like(agents_verts_t)[None].to(self.device, non_blocking=True)  # (1, V, 3)
                agents_verts_rgb[0,:] = torch.Tensor(np.array([255 / 255, 0 / 255, 0 / 255], dtype=np.float32))
                agents_textures = TexturesVertex(verts_features=agents_verts_rgb)
                agents_mesh = Meshes(verts=[agents_verts_t], faces=[agents_faces_t], textures=agents_textures)

                # no history frames
                break

            view_height = 225
            camera_position = torch.Tensor(np.array([center_world[0], center_world[1], view_height])).to(self.device, non_blocking=True)
            r = transforms3d.euler.euler2mat(0, np.pi, yaw)[None]
            R = torch.Tensor(r).to(self.device, non_blocking=True)
            T = -torch.bmm(R.transpose(1, 2), camera_position[None, :, None])[:, :, 0]
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            map_agents_mesh = join_meshes_as_scene([self.map_mesh, agents_mesh])

            images = self.renderer(map_agents_mesh, cameras=cameras).squeeze(0)
        return images

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return in_im
%%writefile ext_ego.py
from typing import Optional, Tuple, cast
import warnings

import numpy as np
import torch

from l5kit.data import DataManager, get_frames_slice_from_scenes
from l5kit.dataset import AgentDataset

class ExtAgentDataset(AgentDataset):
    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame
        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp
        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        try:
            if self.cfg["raster_params"]["disable_traffic_light_faces"]:
                tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        except KeyError:
            warnings.warn(
                "disable_traffic_light_faces not found in config, this will raise an error in the future",
                RuntimeWarning,
                stacklevel=2,
            )
        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)
        # 0,1,C -> C,0,1
        if isinstance(data["image"], torch.Tensor):
            image = data["image"].permute(2, 0, 1)
        else:
            image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "image": image,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "world_to_image": data["world_to_image"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
        }
import os
import torch

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
torch.cuda.is_available()
from pytorch3d_rasterizer import PyTorch3dSemanticRasterizer
from ext_ego import ExtAgentDataset
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = True
cfg = {
    'format_version': 4,
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
        'raster_size': [512, 512],
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
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 0
    },
    
    'sample_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 12,
        'shuffle': False,
        'num_workers': 0
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 10000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
from l5kit.rasterization.rasterizer_builder import (_load_metadata, get_hardcoded_world_to_ecef)

raster_cfg = cfg["raster_params"]
semantic_map_path = dm.require(raster_cfg["semantic_map_key"])
try:
    dataset_meta = _load_metadata(raster_cfg["dataset_meta_key"], dm)
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
except (KeyError, FileNotFoundError):
    world_to_ecef = get_hardcoded_world_to_ecef()

rasterizer = PyTorch3dSemanticRasterizer(
            raster_size=np.array(raster_cfg["raster_size"]),
            pixel_size=np.array(raster_cfg["pixel_size"]),
            ego_center=np.array(raster_cfg["ego_center"]),
            filter_agents_threshold=0.5,
            history_num_frames=0,
            semantic_map_path=semantic_map_path,
            world_to_ecef=world_to_ecef,
            filename="/kaggle/input/lyft-map-mesh/mesh.p"
        )

# Train dataset/dataloader
train_cfg = cfg["sample_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = ExtAgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

print(train_dataset)
%%time
for data in train_dataloader:
    print(data["image"].shape)
    break
plt.figure(figsize=(8, 8))
plt.imshow(data["image"][0].permute(1,2,0).cpu().numpy()[::-1])
cfg = {
    'format_version': 4,
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
        'raster_size': [512, 512],
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
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 2
    },
    
    'sample_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 12,
        'shuffle': False,
        'num_workers': 0
    },
    
    'train_params': {
        'max_num_steps': 100 if DEBUG else 10000,
        'checkpoint_every_n_steps': 5000,
        
        # 'eval_every_n_steps': -1
    }
}
# original case

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
train_cfg = cfg["sample_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

print(train_dataset)
%%time
for data in train_dataloader:
    print(data["image"].shape)
    break
data["image"].shape
plt.figure(figsize=(8, 8))
plt.imshow(data["image"][0][22:].permute(1,2,0).cpu().numpy());
plt.imshow(data["image"][0][0].cpu().numpy(), cmap="Reds", alpha=0.5);
plt.figure(figsize=(8, 8))
plt.imshow(data["image"][0][22:].permute(1,2,0).cpu().numpy());
