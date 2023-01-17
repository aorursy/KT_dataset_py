!pip install lyft-dataset-sdk -q
from datetime import datetime

from functools import partial

import glob

from multiprocessing import Pool



# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces

# even more threads which would lead to a lot of context switching, slowing things down a lot.

import os

os.environ["OMP_NUM_THREADS"] = "1"



import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

import cv2

from PIL import Image

import numpy as np

from tqdm import tqdm, tqdm_notebook

import scipy

import scipy.ndimage

import scipy.special

from scipy.spatial.transform import Rotation as R

from pathlib import Path



from lyft_dataset_sdk.lyftdataset import LyftDataset,LyftDatasetExplorer

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion

from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

import time

from lyft_dataset_sdk.utils.map_mask import MapMask
!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_lidar lidar
class LyftTestDataset(LyftDataset):

    """Database class for Lyft Dataset to help query and retrieve information from the database."""



    def __init__(self, data_path: str, json_path: str, verbose: bool = True, map_resolution: float = 0.1):

        """Loads database and creates reverse indexes and shortcuts.

        Args:

            data_path: Path to the tables and data.

            json_path: Path to the folder with json files

            verbose: Whether to print status messages during load.

            map_resolution: Resolution of maps (meters).

        """



        self.data_path = Path(data_path).expanduser().absolute()

        self.json_path = Path(json_path)



        self.table_names = [

            "category",

            "attribute",

            "sensor",

            "calibrated_sensor",

            "ego_pose",

            "log",

            "scene",

            "sample",

            "sample_data",

            "map",

        ]



        start_time = time.time()



        # Explicitly assign tables to help the IDE determine valid class members.

        self.category = self.__load_table__("category")

        self.attribute = self.__load_table__("attribute")

        

        

        self.sensor = self.__load_table__("sensor")

        self.calibrated_sensor = self.__load_table__("calibrated_sensor")

        self.ego_pose = self.__load_table__("ego_pose")

        self.log = self.__load_table__("log")

        self.scene = self.__load_table__("scene")

        self.sample = self.__load_table__("sample")

        self.sample_data = self.__load_table__("sample_data")

        

        self.map = self.__load_table__("map")



        # Initialize map mask for each map record.

        for map_record in self.map:

            map_record["mask"] = MapMask(self.data_path / map_record["filename"], resolution=map_resolution)



        if verbose:

            for table in self.table_names:

                print("{} {},".format(len(getattr(self, table)), table))

            print("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))



        # Initialize LyftDatasetExplorer class

        self.explorer = LyftDatasetExplorer(self)

        # Make reverse indexes for common lookups.

        self.__make_reverse_index__(verbose)

        

    def __make_reverse_index__(self, verbose: bool) -> None:

        """De-normalizes database to create reverse indices for common cases.

        Args:

            verbose: Whether to print outputs.

        """



        start_time = time.time()

        if verbose:

            print("Reverse indexing ...")



        # Store the mapping from token to table index for each table.

        self._token2ind = dict()

        for table in self.table_names:

            self._token2ind[table] = dict()



            for ind, member in enumerate(getattr(self, table)):

                self._token2ind[table][member["token"]] = ind



        # Decorate (adds short-cut) sample_data with sensor information.

        for record in self.sample_data:

            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])

            sensor_record = self.get("sensor", cs_record["sensor_token"])

            record["sensor_modality"] = sensor_record["modality"]

            record["channel"] = sensor_record["channel"]



        # Reverse-index samples with sample_data and annotations.

        for record in self.sample:

            record["data"] = {}

            record["anns"] = []



        for record in self.sample_data:

            if record["is_key_frame"]:

                sample_record = self.get("sample", record["sample_token"])

                sample_record["data"][record["channel"]] = record["token"]



        # Add reverse indices from log records to map records.

        if "log_tokens" not in self.map[0].keys():

            raise Exception("Error: log_tokens not in map table. This code is not compatible with the teaser dataset.")

        log_to_map = dict()

        for map_record in self.map:

            for log_token in map_record["log_tokens"]:

                log_to_map[log_token] = map_record["token"]

        for log_record in self.log:

            log_record["map_token"] = log_to_map[log_record["token"]]



        if verbose:

            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))
level5data = LyftTestDataset(data_path='.', json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/test_data', verbose=True)

# Our code will generate data, visualization and model checkpoints, they will be persisted to disk in this folder

ARTIFACTS_FOLDER = "./artifacts"

os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
sample_sub = pd.read_csv('../input/3d-object-detection-for-autonomous-vehicles/sample_submission.csv')

sample_sub.head()
def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):

    """

    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.

    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.

    

    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.

    """

    

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    

    tm = np.eye(4, dtype=np.float32)

    translation = shape/2 + offset/voxel_size

    

    tm = tm * np.array(np.hstack((1/voxel_size, [1])))

    tm[:3, 3] = np.transpose(translation)

    return tm



def transform_points(points, transf_matrix):

    """

    Transform (3,N) or (4,N) points using transformation matrix.

    """

    if points.shape[0] not in [3,4]:

        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))

    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]
def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):

    if len(shape) != 3:

        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

        

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:

        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))



    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))

    p = transform_points(points, tm)

    return p



def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):



    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)

    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)

    points_voxel_coords = np.int0(points_voxel_coords)

    

    bev = np.zeros(shape, dtype=np.float32)

    bev_shape = np.array(shape)



    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    

    points_voxel_coords = points_voxel_coords[within_bounds]

    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

        

    # Note X and Y are flipped:

    bev[coord[:,1], coord[:,0], coord[:,2]] = count

    

    return bev



def normalize_voxel_intensities(bev, max_intensity=16):

    return (bev/max_intensity).clip(0,1)
bev_shape = (336, 336, 3)

target_im = np.zeros(bev_shape, dtype=np.uint8)



def move_boxes_to_car_space(boxes, ego_pose):

    """

    Move boxes from world space to car space.

    Note: mutates input boxes.

    """

    translation = -np.array(ego_pose['translation'])

    rotation = Quaternion(ego_pose['rotation']).inverse

    

    for box in boxes:

        # Bring box to car space

        box.translate(translation)

        box.rotate(rotation)

        

def scale_boxes(boxes, factor):

    """

    Note: mutates input boxes

    """

    for box in boxes:

        box.wlh = box.wlh * factor



def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):

    for box in boxes:

        # We only care about the bottom corners

        corners = box.bottom_corners()

        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)

        corners_voxel = corners_voxel[:,:2] # Drop z coord



        class_color = classes.index(box.name) + 1

        

        if class_color == 0:

            raise Exception("Unknown class: {}".format(box.name))



        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)
def visualize_lidar_of_sample(sample_token, axes_limit=80):

    sample = level5data.get("sample", sample_token)

    sample_lidar_token = sample["data"]["LIDAR_TOP"]

    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)

    

# Don't worry about it being mirrored.

visualize_lidar_of_sample(sample_sub.loc[0,'Id'])
# Some hyperparameters we'll need to define for the system

voxel_size = (0.4, 0.4, 1.5)

z_offset = -2.0

bev_shape = (336, 336, 3)



# We scale down each box so they are more separated when projected into our coarse voxel space.

box_scale = 0.8



NUM_WORKERS = os.cpu_count() * 3



# "bev" stands for birds eye view

# test_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_test_data")

test_data_folder = '/kaggle/working/artifacts'
def prepare_testing_data_for_scene(sample_token, output_folder=test_data_folder,

                                   bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset,

                                   box_scale=box_scale):

    """

    Given a sample token (in a scene), output rasterized input volumes in birds-eye-view perspective.



    """

    

#     while sample_token:

        

    sample = level5data.get("sample", sample_token)

    



    sample_lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_data = level5data.get("sample_data", sample_lidar_token)

    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

    

    



    ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])

    calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    





    global_from_car = transform_matrix(ego_pose['translation'],

                                       Quaternion(ego_pose['rotation']), inverse=False)

    



    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),

                                        inverse=False)

    

    

    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

    

    lidar_pointcloud.transform(car_from_sensor)



    bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)

    bev = normalize_voxel_intensities(bev)



    bev_im = np.round(bev*255).astype(np.uint8)



    cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
# # to debug..

# # doesn't work right now

# os.makedirs(test_data_folder, exist_ok=True)

# # NUM_WORKERS = 4

# pool = Pool(NUM_WORKERS)

# for _ in tqdm_notebook(pool.imap(partial(prepare_testing_data_for_scene,

#                        output_folder=test_data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale),sample_sub.loc[:,'Id'].values)

#                       ,total=len(sample_sub)):

#     pass



# pool.close()

# del pool
for token in tqdm_notebook(sample_sub.loc[:,'Id'].values):

    prepare_testing_data_for_scene(token)
!tar -czf lyft3d_bev_test_data.tar.gz ./artifacts/ 
!du -h lyft3d_bev_test_data.tar.gz
!rm -r ./artifacts