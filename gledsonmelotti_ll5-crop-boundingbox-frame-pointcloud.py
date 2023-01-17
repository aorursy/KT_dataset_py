print('From LL5')
print('Bounding Box-RGB Image')


import os
import gc
import numpy as np
import pandas as pd

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
from shapely.geometry import MultiPoint, box
from pyquaternion import Quaternion
from tqdm import tqdm


from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud, RadarPointCloud  # NOQA
from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix

from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy
import scipy.io as sio


###############################################################################
# box_vis_level: BoxVisibility = BoxVisibility.ANY  # Requires at least one corner visible in the image.
box_vis_level: BoxVisibility = BoxVisibility.ALL  # Requires all corners are inside the image.

def post_process_coords(corner_coords, w, h):
    imsize=(w,h)
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return []

def LL5_render_annotation(my_annotation_token, w, h, view = np.eye(4),box_vis_level=box_vis_level):
    ann_token=my_annotation_token
    w=w
    h=h
    """Render selected annotation.

    Args:
        ann_token: Sample_annotation token.
        margin: How many meters in each direction to include in LIDAR view.
        view: LIDAR view point.
        box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        out_path: Optional path to save the rendered figure to disk.

    """

    ann_record = lyftd.get("sample_annotation", ann_token)
    categoria=ann_record['category_name']
    sample_record = lyftd.get("sample", ann_record["sample_token"])
    lidar_top = [key for key in sample_record["data"].keys() if "LIDAR_TOP" in key]
    if lidar_top==['LIDAR_TOP']:
        boxes_cam, cam, boxes_lidar = [], [], []
        cams = [key for key in sample_record["data"].keys() if "CAM" in key]
        camera=[]
        for cam in cams:
            data_path_cam, boxes_cam, camera_intrinsic = lyftd.get_sample_data(
                sample_record["data"][cam], box_vis_level=box_vis_level, selected_anntokens=[ann_token])
            if len(boxes_cam) > 0:
                break  # We found an image that matches. Let's abort.
        if len(boxes_cam)==1:
            camera=cam
            cam = sample_record["data"][cam]
                    # CAMERA view
            path_cam, boxes_cam, camera_intrinsic = lyftd.get_sample_data(cam, selected_anntokens=[ann_token])
            data_path_cam=path_cam.parts[5]
            corners_camera3D=[]
            corners_camera2D=[]
            for f in range(len(boxes_cam)):
                 corners_camera3D=boxes_cam[f].corners().T
                 corners_3d = boxes_cam[f].corners()

                # Project 3d box to 2d.
                 corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
    
                # Keep only corners that fall within the image.
                 final_coords = post_process_coords(corner_coords, w, h)
                 #min_x, min_y, max_x, max_y = final_coords
                 corners_camera2D=final_coords
                 
            # LIDAR view
            lidar = sample_record["data"]["LIDAR_TOP"]
            path_lidar, boxes_lidar, lidar_intrinsic = lyftd.get_sample_data(lidar, selected_anntokens=[ann_token])
            data_path_lidar=path_lidar.parts[5]
            pc=LidarPointCloud.from_file(path_lidar)
            pointclouds = pc.points.T
            corners_lidar=[]
            #for box in boxes_lidar:
            for v in range(len(boxes_lidar)):
                boxes3D=boxes_lidar[v].corners()
                corners_lidar=boxes3D.T
    
            return data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria
        else:
            b='boxes_cam vazio'
            print(b)
            data_path_cam=[]
            camera=[] 
            corners_camera3D=[]
            corners_camera2D=[]
            data_path_lidar=[]
            corners_lidar=[]
            pointclouds=[]
            categoria=[]
            return data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria
    else:
         a='lidar_top é diferente de LIDAR_TOP'
         print(a)

DATA_PATH = 'D:/Dataset_LyftLevel5/Perception/train/'

lyftd = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')
lyftd.category
lyftd.attribute
attribute=[]
attribute1=lyftd.attribute
for u in range(len(attribute1)):
    attribute.append(attribute1[u]['token'])

total_scene = lyftd.scene

data_path_cam_final=[]
camera_final=[]
boxes_cam_final3D=[]
camera_intrinsic_final=[]
data_path_lidar_final=[]
boxes_lidar_final=[]
categoria_final=[]
pointclouds_final=[]
boxes_cam_final2D=[]


j=1 # Odd Frames; j = 0 Even Frames 
for jj in range(len(total_scene)):
    print('Impar:', j)
    my_scene = lyftd.scene[j]
    my_sample_token_first = my_scene["first_sample_token"]
    my_sample_first = lyftd.get('sample', my_sample_token_first)
    lidar_top_first = my_sample_first['data']['LIDAR_TOP']
    first_token=my_sample_token_first
    my_sample_token_last = my_scene["last_sample_token"]
    my_sample_last = lyftd.get('sample', my_sample_token_last)
    lidar_top_last = my_sample_last['data']['LIDAR_TOP']
    last_token=my_sample_token_last
    next_token=my_sample_token_first
    nexttoken=[]
    while next_token!='':
      nexttoken.append(next_token)
      next_token=lyftd.get('sample', next_token)['next']
    next_flip=nexttoken
    first_next_last_token= nexttoken
    for v in range(len(first_next_last_token)):
        sensor_channel = 'CAM_FRONT'
        my_sampleLL5 = lyftd.get('sample', first_next_last_token[v])
        my_sampleLL5_data = lyftd.get('sample_data', my_sampleLL5['data'][sensor_channel])
        w=my_sampleLL5_data['width']
        h=my_sampleLL5_data['height']
        for k in range(len(my_sampleLL5['anns'])):
            print(str(j)+'_'+str(v)+'_'+str(k))
            my_annotation_token = my_sampleLL5['anns'][k]
            box_vis_level: BoxVisibility = BoxVisibility.ALL
            data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria = LL5_render_annotation(my_annotation_token, w, h, view = np.eye(4), box_vis_level=box_vis_level)
            if len(data_path_cam)!=0 and len(camera)!=0 and len(corners_camera3D)!=0 and len(corners_camera2D)!=0 and len(data_path_lidar)!=0 and len(corners_lidar)!=0 and len(pointclouds)!=0 and len(categoria)!=0:
                if w==1920 and h==1080:
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/data_path_cam/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_cam':data_path_cam})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/camera/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'camera':camera})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/corners_camera3D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera3D':corners_camera3D})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/corners_camera2D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera2D':corners_camera2D})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/data_path_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_lidar':data_path_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/corners_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_lidar':corners_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/pointclouds/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'pointclouds':pointclouds})
                    sio.savemat('D:/Dataset_LyftLevel5/1920_impar/categoria/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'categoria':categoria})
                elif w==1224 and h==1024:
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/data_path_cam/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_cam':data_path_cam})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/camera/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'camera':camera})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/corners_camera3D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera3D':corners_camera3D})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/corners_camera2D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera2D':corners_camera2D})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/data_path_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_lidar':data_path_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/corners_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_lidar':corners_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/pointclouds/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'pointclouds':pointclouds})
                    sio.savemat('D:/Dataset_LyftLevel5/1224_impar/categoria/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'categoria':categoria})
    j=j+2
print("From LL5")
print('Bounding Box-Point Cloud')

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from matplotlib.axes import Axes
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

import scipy.io as sio

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import Box, RadarPointCloud, LidarPointCloud  # NOQA
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
from lyft_dataset_sdk.utils.map_mask import MapMask

# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
level5data = LyftDataset(data_path='D:/Dataset_LyftLevel5/Perception/train', json_path='D:/Dataset_LyftLevel5/Perception/train/train_data', verbose=True)


#ALL = 0  # Requires all corners are inside the image.
#ANY = 1  # Requires at least one corner visible in the image.
#NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.

#######################
import pandas as pd
train = pd.read_csv('D:/Dataset_LyftLevel5/Perception/train/' + 'train.csv')
# Taken from https://www.kaggle.com/gaborfodor/eda-3d-object-detection-challenge

object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',
                  'width', 'length', 'height', 'yaw', 'class_name']
objects = []
for sample_id, ps in tqdm(train.values[:]):
    object_params = ps.split()
    n_objects = len(object_params)
    for i in range(n_objects // 8):
        x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
        objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
train_objects = pd.DataFrame(
    objects,
    columns = object_columns
)

numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)
train_objects.head()
########################

#class LidarPointCloud(LidarPointCloud):
#    @staticmethod
#    def nbr_dims() -> int:
#        """Returns the number of dimensions.
#
#        Returns: Number of dimensions.
#
#        """
#        return 4
#
#    @classmethod
#    def from_file(cls, file_name: Path) -> "LidarPointCloud":
#        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
#
#        Args:
#            file_name: Path of the pointcloud file on disk.
#
#        Returns: LidarPointCloud instance (x, y, z, intensity).
#
#        """
#
#        assert file_name.suffix == ".bin", "Unsupported filetype {}".format(file_name)
#
#        scan = np.fromfile(str(file_name), dtype=np.float32)
#        if len(scan) % 5 == 0:
#            points = scan.reshape((-1, 5))[:, : cls.nbr_dims()]
#        else:
#            scan0=[]
#            scan1=[]
#            scan2=[]
#            scan3=[]
#            scan4=[]
#            scan_pontos = []
#            scan_pontos = scan.reshape((-1, 1))
#            dv = 0
#            for divisao in range(math.floor(len(scan)/5)):
#                if (dv+4)<=len(scan):
#                    scan0.append(scan_pontos[dv])
#                    scan1.append(scan_pontos[dv+1])
#                    scan2.append(scan_pontos[dv+2])
#                    scan3.append(scan_pontos[dv+3])
#                    scan4.append(scan_pontos[dv+4])
#                dv = dv+5
#            points = np.concatenate((scan0,scan1,scan2,scan3),axis=1)
#        return cls(points.T)


#def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
#    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
#    orthographic projections. It first applies the dot product between the points and the view. By convention,
#    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
#    normalization along the third dimension.
#
#    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
#    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
#    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
#     all zeros) and normalize=False
#
#    Args:
#        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
#        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
#        The projection should be such that the corners are projected onto the first 2 axis.
#        normalize: Whether to normalize the remaining coordinate (along the third axis).
#
#    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
#
#    """
#
#    assert view.shape[0] <= 4
#    assert view.shape[1] <= 4
#    assert points.shape[0] == 3
#
#    viewpad = np.eye(4)
#    viewpad[: view.shape[0], : view.shape[1]] = view
#
#    nbr_points = points.shape[1]
#
#    # Do operation in homogenous coordinates
#    points = np.concatenate((points, np.ones((1, nbr_points))))
#    points = np.dot(viewpad, points)
#    points = points[:3, :]
#
#    if normalize:
#        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
#
#    return points

def LL5_map_pointcloud_to_image(pointsensor_token, camera_token):
    """Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to
    the image plane.

    Args:
        pointsensor_token: Lidar/radar sample_data token.
        camera_token: Camera sample_data token.

    Returns: (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).

    """

    cam = level5data.get("sample_data", camera_token)
    nome_cam=Path(cam["filename"]).parts
    nome_cam_part=nome_cam[1]
    nome_cam_size=len(nome_cam_part)
    nome_cam_final=nome_cam_part[0:nome_cam_size-5]
    pointsensor = level5data.get("sample_data", pointsensor_token)
    pcl_path = level5data.data_path / pointsensor["filename"]

    if pointsensor["sensor_modality"] == "lidar":
        pc = LidarPointCloud.from_file((pcl_path))
    else:
        pc = RadarPointCloud.from_file((pcl_path))
    image = Image.open(str(level5data.data_path / cam["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)    
    points = points[:, mask]
    coloring = coloring[mask]
    
    auxiliar = pc.points
    auxiliar = auxiliar[:, mask]
    auxiliar = np.delete(auxiliar, 0, 0)
    auxiliar = np.delete(auxiliar, 0, 0)
    points = np.delete(points, 2, 0)
    points = np.concatenate((points,auxiliar),axis=0)
    
    return points, coloring, image, nome_cam_final

#level5data.list_scenes()

for k in range(0,180):
    my_scene = level5data.scene[k]
    print(k)
    name = my_scene["name"]
    nbr_samples = my_scene['nbr_samples']
    for j in range(0,nbr_samples):
        if j==0:
            my_sample_token = my_scene["first_sample_token"]
            #level5data.render_sample(my_sample_token)
            my_sample = level5data.get('sample', my_sample_token)
            #my_sample
            #level5data.list_sample(my_sample['token'])
            codigo_token = my_sample['token']
        else:
            my_sample = level5data.get('sample', codigo_token)
            codigo_token = my_sample["next"]
            
        sample_token = codigo_token
        sample_record = level5data.get("sample", sample_token)
        pointsensor_channel = "LIDAR_TOP"
        #camera = ["CAM_FRONT_ZOOMED"]
        camera = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_FRONT_ZOOMED","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
        # Here we just grab the front camera and the point sensor.
        for cam in range(0,len(camera)):
            camera_channel = camera[cam]
            pointsensor_token = sample_record["data"][pointsensor_channel]
            camera_token = sample_record["data"][camera_channel]
            points, coloring, image, nome_cam_final = LL5_map_pointcloud_to_image(pointsensor_token, camera_token)
            # Here we just grab the front camera and the point sensor.
    #        plt.figure(figsize=(9, 16))
    #        plt.imshow(im)
    #        plt.scatter(points[0, :], points[1, :], c=coloring, s=1)
    #        plt.axis("off")
            im = np.array(image)
#            if (len(im[:,0])==1080 and len(im[0,:])==1920):
#                name_final = str(k)+'_'+str(j)+'_'+name
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Image/'+name_final+'.mat',{'im':im})
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Projected/'+name_final+'.mat',{'points':points})
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Coloring/'+name_final+'.mat',{'coloring':coloring})

            #name_final = str(k)+'_'+str(j)+'_'+camera_channel
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Image/'+nome_cam_final+'.mat',{'im':im})
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Projected/'+nome_cam_final+'.mat',{'points':points})
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Coloring/'+nome_cam_final+'.mat',{'coloring':coloring})