# Operating system

import sys

import os

from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"



# math

import numpy as np

from numpy import arange

import math

from numpy import linalg as LA

from scipy import stats

#progress bar

from tqdm import tqdm, tqdm_notebook

tqdm.pandas()

from datetime import timezone, datetime, timedelta



# data analysis

import pandas as pd



#plotting

import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from matplotlib import animation, rc

import matplotlib.colors as colors



#plotting 3d

import plotly.graph_objects as go





#machine learning

import sklearn

import h5py

import sklearn.metrics

from sklearn.model_selection import train_test_split

from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from sklearn.preprocessing import StandardScaler

# Lyft dataset SDK

!pip install lyft-dataset-sdk

from lyft_dataset_sdk.utils.map_mask import MapMask

from lyft_dataset_sdk.lyftdataset import LyftDataset

from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion



DATA_PATH = './'
DEBUG = True

def log(message):

    if(DEBUG == True):

        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')

        print(time_string + ' : ', message )

os.system('rm -f data && ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')

os.system('rm  -f images && ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')

os.system('rm  -f maps && ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')

os.system('rm  -f lidar && ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
LYFT = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH + 'data', verbose=True)
# Examine object sizes

wlhs = np.array([ann['size'] for ann in  LYFT.sample_annotation])

# max, min  height

np.max(wlhs[:,2]), np.min(wlhs[:,2])

# (8.862, 0.333)
# max, min  length

np.max(wlhs[:,1]), np.min(wlhs[:,1])

#  (22.802, 0.261)
# max, min  width

np.max(wlhs[:,0]), np.min(wlhs[:,0])

# (4.157, 0.223)
# To unzip lidar files when needed (not needed for Kaggle execution)

def unzip(row, mode='train'):

    zip_command = "unzip ../3d-object-detection-for-autonomous-vehicles.zip " + mode + '_' + row['filename'].astype(str)           + " -d "            + CWD + "/../data/"        

    os.system(zip_command)    



def removefile(row, mode='train'):

    rm_command = "rm -f  " + CWD + "/../data/"+ mode + '_' + row['filename'].astype(str)                  

    os.system(zip_command)    



def sort_points_by_coord(points, coord):

    indices = np.argsort(points[:,coord],axis=0).reshape(-1,1)

    indices = np.repeat(indices, points.shape[-1],axis=-1)

    # print(indices)

    sorted = np.take_along_axis(points,indices,axis=0)

    del indices

    return sorted



def sort_points(points):

    sorted_points = sort_points_by_coord(points, 2)

    sorted_points = sort_points_by_coord(sorted_points, 1)

    return sort_points_by_coord(sorted_points, 0)



    

# https://www.geeksforgeeks.org/linear-regression-python-implementation/

def estimate_regression_coef(points): 



    x = points[:,0]

    y = points[:,1]

    # number of observations/points 

    n = np.size(x) 

  

    # mean of x and y vector 

    m_x, m_y = np.mean(x), np.mean(y) 

  

    # calculating cross-deviation and deviation about x 

    SS_xy = np.sum(y*x) - n*m_y*m_x 

    SS_xx = np.sum(x*x) - n*m_x*m_x 

  

    # calculating regression coefficients 

    b_1 = SS_xy / SS_xx 

    b_0 = m_y - b_1*m_x 

  

    return b_1 



def length_of_xy_diagonal(points):

    x = points[:,0]

    y = points[:,1]

    return LA.norm([np.max(x)- np.min(x), np.max(y)- np.min(y)])



def length_of_xz_diagonal(points):

    x = points[:,0]

    z = points[:,2]

    return LA.norm([np.max(x)- np.min(x), np.max(z)- np.min(z)])





def slope(points):

    x = points[:,0]

    y = points[:,1]

    m, _, _, _, _ = stats.linregress(x, y)

    return m



def yaw(points):

#     b1 = estimate_regression_coef (points)

    b1 = slope(points)

    if b1 == 0:

        return math.pi / 2

    else:

        return np.arctan(1/b1)



def rotation_matrix_xy(theta):

     return np.array([ 

        [math.cos(theta), -math.sin(theta), 0],

        [math.sin(theta), math.cos(theta), 0],

        [0, 0, 1],

    ])



# Rotate point cluster using yaw -> calclulate min, max and construct boxs -> inverse rotate to world coords

def get_min_bbox_corners(candidate, yw):

    xyz = np.delete(candidate, np.s_[3], axis=1) 

    world_to_candidate_rotation_matrix = rotation_matrix_xy(-yw)

    rotated = np.matmul(xyz, world_to_candidate_rotation_matrix)

    minx = np.min(rotated[:,0])

    maxx = np.max(rotated[:,0])

    miny = np.min(rotated[:,1])

    maxy = np.max(rotated[:,1])

    minz = np.min(rotated[:,2])

    maxz = np.max(rotated[:,2])

    corners_rotated = np.array([

        [minx, miny, minz],

        [maxx, miny, minz],

        [maxx, miny, maxz],

        [minx, miny, maxz],               

        [minx, maxy, minz],

        [maxx, maxy, minz],

        [maxx, maxy, maxz],

        [minx, maxy, maxz],               

    ])

    del xyz

    del rotated

    corners = np.matmul(corners_rotated, world_to_candidate_rotation_matrix.T)

    return (corners_rotated, corners)



def get_centroid(corners):

    minx = np.min(corners[:,0])

    maxx = np.max(corners[:,0])

    miny = np.min(corners[:,1])

    maxy = np.max(corners[:,1])

    minz = np.min(corners[:,2])

    maxz = np.max(corners[:,2])

    x = (minx + maxx) / 2

    y = (miny + maxy) / 2

    z = (minz + maxz) / 2

    return x, y, z



def get_dimensions(corners):

    minx = np.min(corners[:,0])

    maxx = np.max(corners[:,0])

    miny = np.min(corners[:,1])

    maxy = np.max(corners[:,1])

    minz = np.min(corners[:,2])

    maxz = np.max(corners[:,2])

    w = (maxx - minx) 

    l = (maxy - miny) 

    h = (maxz - minz) 

    return w, l, h







# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/



def minmax(box):

    minx = np.min(box[:,0])

    maxx = np.max(box[:,0])

    miny = np.min(box[:,1])

    maxy = np.max(box[:,1])

    minz = np.min(box[:,2])

    maxz = np.max(box[:,2])    

    return minx, maxx, miny , maxy, minz, maxz

    

def intersection(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle

    minxa, maxxa, minya, maxya, minza, maxza = minmax(boxA)

    minxb, maxxb, minyb, maxyb, minzb, maxzb = minmax(boxB)



    

    xA = max(minxa, minxb)

    yA = max(minya, minyb)

    zA = max(minza, minzb)

    xB = min(maxxa, maxxb)

    yB = min(maxya, maxyb)

    zB = min(maxza, maxzb)



    # compute the volume of intersection cuboid

    intersectionVolume = max(0, xB - xA ) * max(0, yB - yA )  * max(0, zB - zA) 

    return intersectionVolume



def union(boxA, boxB):

    aw, al, ah = get_dimensions(boxA)

    bw, bl, bh = get_dimensions(boxB)





    totalVolume = aw * al * ah + bw * bl * bh - intersection(boxA, boxB)

    return totalVolume



def iou(boxA, boxB):

    return intersection(boxA,boxB) / union(boxA, boxB)



def make_a_box(minx, maxx, miny, maxy, minz,maxz):

    return np.array([

        [minx, miny, minz],

        [maxx, miny, minz],

        [maxx, miny, maxz],

        [minx, miny, maxz],               

        [minx, maxy, minz],

        [maxx, maxy, minz],

        [maxx, maxy, maxz],

        [minx, maxy, maxz],               

    ])
# Unit test IOU



boxA = make_a_box(2,3,2,3,2,3)

boxB = make_a_box(1,5,1,5,1,5)





boxB = np.array([[2174.97305545, 988.72661905,  -17.47796403],

 [2173.87907608,  986.95519783 , -17.47796403],

 [2173.87907608,  986.95519783 , -19.45096403],

 [2174.97305545,  988.72661905 , -19.45096403],

 [2170.5096185 ,  991.48311077 , -17.47796403],

 [2169.41563913,  989.71168955 , -17.47796403],

 [2169.41563913 , 989.71168955 , -19.45096403],

 [2170.5096185 ,  991.48311077,  -19.45096403]] )

boxA = np.array( [[2173.90940966 , 987.70527066,  -18.97512237],

 [2174.19097651 , 987.47654011,  -18.97512237],

 [2174.19097651 , 987.47654011,  -18.40049185],

 [2173.90940966 , 987.70527066,  -18.40049185],

 [2174.39607111 , 988.30434993,  -18.97512237],

 [2174.67763796 , 988.07561938,  -18.97512237],

 [2174.67763796,  988.07561938,  -18.40049185],

 [2174.39607111,  988.30434993,  -18.40049185]])



#intersection(boxA, boxB)

#minmax(boxA), minmax(boxB)

iou(boxA, boxB), union(boxA, boxB), intersection(boxA, boxB)
# Extract composite dataframe for list of sample tokens

def extract_data_for_clustering(tokens):

    sampledata_df = pd.DataFrame(LYFT.sample_data)

    sampledata_df = sampledata_df[sampledata_df['sample_token'].isin(tokens)]

    sampledata_df = sampledata_df[sampledata_df['fileformat'] == 'bin']

    sampledata_df.rename(columns={'token':'sampledata_token'}, inplace=True)

    sampledata_df = sampledata_df[[

        'sample_token', 

        'sampledata_token',

        'ego_pose_token', 

        'channel',

        'calibrated_sensor_token',

        'fileformat',

        'filename']]



    ep_df = pd.DataFrame(LYFT.ego_pose)

    ep_df.rename(columns={'token':'ego_pose_token', 'rotation': 'ep_rotation', 'translation': 'ep_translation'}, inplace=True)

    ep_df = ep_df[['ego_pose_token',

                     'ep_rotation',

                     'ep_translation']]

    sampledata_df = pd.merge(sampledata_df, ep_df, left_on='ego_pose_token', right_on='ego_pose_token',how='inner')





    cs_df = pd.DataFrame(LYFT.calibrated_sensor)

    cs_df.rename(columns={'token':'calibrated_sensor_token', 'rotation': 'cs_rotation', 'translation': 'cs_translation'}, inplace=True)

    cs_df = cs_df[['calibrated_sensor_token',\

                     'cs_rotation',\

                     'cs_translation',\

                     'camera_intrinsic'

                  ]]

    sampledata_df = pd.merge(sampledata_df, cs_df, left_on='calibrated_sensor_token', right_on='calibrated_sensor_token',how='inner')

    # sampledata_df['filepath'] = sampledata_df.apply(lambda row: LYFT.get_sample_data_path(row['sampledata_token']), axis=1)

    sampledata_df['pointcloud'] = sampledata_df.apply(lambda row: LidarPointCloud.from_file(LYFT.get_sample_data_path(row['sampledata_token'])).points, axis=1)

    sampledata_df = sampledata_df[[

        'sample_token', 

        'sampledata_token',

        'ep_rotation',

        'ep_translation',

        'channel',

        'cs_rotation',

        'cs_translation',

        # 'filepath',

        'pointcloud',

        'fileformat',

        'filename']]

    



    return sampledata_df.copy(deep=True)
def car_to_world_points_notf (points, translation, rotation):

    rotated = np.dot(Quaternion(rotation).rotation_matrix, points.T)

    translated = np.add(rotated.T, translation)

    return translated



def sensor_to_car_points_notf (points, translation, rotation):

    rotated = np.dot(Quaternion(rotation).rotation_matrix, points.T)

    translated = np.add(rotated.T, translation)

    return translated



# Take all LIDAR points for a sample and merge them in world coordinates

def get_lidar_points_for_clustering(sample_token, token_input, data_path):

    pt_cloud = np.zeros((0,3))

    ground_z = -20.0

    for i in range(len(token_input)):



        sampledatarow = token_input.iloc[i]

        cs_t = sampledatarow['cs_translation']

        cs_r = sampledatarow['cs_rotation']

        ep_t = sampledatarow['ep_translation']

        ep_r = sampledatarow['ep_rotation']

        ground_z = min(ground_z, ep_t[2])

        ego_x = ep_t[0]

        ego_y = ep_t[1]





        pointcloud = sampledatarow['pointcloud']

        # pointcloud = get_lidar_pointcloud_for_clustering(sampledatarow['filepath'])

        pc_points_t = pointcloud.T

        pc_points_t = pc_points_t[:,:3]

        pc_points_t = sensor_to_car_points_notf(pc_points_t, cs_t, cs_r)



        pc_points_t = car_to_world_points_notf(pc_points_t, ep_t, ep_r)

        print(pc_points_t.shape, ground_z, ep_t[2],np.min(pc_points_t[:,2]), np.max(pc_points_t[:,2]))

        pc_points_t = pc_points_t[pc_points_t[:,0] < (ego_x+100)]

        pc_points_t = pc_points_t[pc_points_t[:,0] > (ego_x-100)]



        pc_points_t = pc_points_t[pc_points_t[:,1] < (ego_y+100)]

        pc_points_t = pc_points_t[pc_points_t[:,1] > (ego_y-100)]





        pc_points_t = pc_points_t[pc_points_t[:,2] < (ground_z+5.3)]

        pc_points_t = pc_points_t[pc_points_t[:,2] > (ground_z+1)]

        print(pc_points_t.shape,  ground_z,ep_t[2], np.min(pc_points_t[:,2]), np.max(pc_points_t[:,2]))





        pt_cloud = np.concatenate([pt_cloud, pc_points_t], axis=0)

        #del pointcloud

        #del pc_points_t

    return ground_z,  pt_cloud      
# Given a sample token, identify point clusters with bounding box dimensions for that sample

# Uses DBSCAN clustering

def identify_clusters(sample_token, token_input, data_path, eps=0.3, min_samples=10):

    yaw_list = []

    corners_list = []

    dimensions_list = []

    xyzs = np.zeros((0,3))

    wlhs = np.zeros((0,3))

    xs = []

    ys = []

    zs = []

    ws = []

    ls = []

    hs = []

    equisized = []

    has_rows = False



    log('getting lidar points' + sample_token)

    ground_z, all_points = get_lidar_points_for_clustering(sample_token, token_input, data_path)

    log('got lidar points' + sample_token)



    if(all_points.shape[0] > 0):

        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(all_points)

        log('clustered' + sample_token)

        labels = clusters.labels_

        unique_labels = set(labels)

        points_with_labels = np.concatenate((all_points, labels.reshape((-1,1))), axis=1)

        noise_points = labels == -1

        points_with_labels = points_with_labels[~noise_points]

        if(points_with_labels.shape[0] > 0):

            ulabels = [x for x in unique_labels if x != -1]

            # ulabels = unique_labels[unique_labels != -1]

            if(len(ulabels) > 0):

                pt_clouds = []

                max_points = 0



                for ulabel in list(ulabels):

                    box_pt_cloud = points_with_labels[points_with_labels[:,3] == ulabel]



                    if(box_pt_cloud.shape[0] == 0):

                        continue

#                     if(length_of_xy_diagonal(box_pt_cloud) > 26):

#                         continue;

                    if(length_of_xz_diagonal(box_pt_cloud) > 10):

                        continue;



                    yw = yaw(box_pt_cloud)

                    corners_rotated, corners = get_min_bbox_corners(box_pt_cloud, yw)

                    if (corners[0][2] > ground_z + 2):

                        continue

                    w, l, h = get_dimensions(corners_rotated)

                    if (w > 5):

                        continue

                    if (l > 25):

                        continue

                    if (h > 5):

                        continue

                    has_rows = True

                    yaw_list.append(yw)

                    corners_list.append(corners)

                    dimensions_list.append((w,l,h))

                    box_pt_cloud = np.delete(box_pt_cloud, np.s_[3], axis=1)



                    num_points = box_pt_cloud.shape[0]



                    if num_points > max_points:

                        max_points = num_points

                    # box_pt_cloud = box_pt_cloud.astype('f')

                    pt_clouds.append((box_pt_cloud))

                if(len(pt_clouds) > 0):

                    for cloud in pt_clouds:

                        ones = np.ones((cloud.shape[0])).reshape(-1,1)

                        cloud = np.concatenate((cloud, ones), axis=1)

                        zeros = np.zeros((max_points - cloud.shape[0],4))

                        fullsize = np.c_[cloud.T, zeros.T].T

                        fullsize = fullsize.astype('f')

                        equisized.append(fullsize)

#                     del pt_clouds

#                     del points_with_labels

#                     del clusters

                    xyzs = np.array([ get_centroid(c) for c in corners_list])

                    wlhs = np.array(dimensions_list)

    log('proposals ready' + sample_token)

        

    proposal_df = pd.DataFrame()

    proposal_df['yaw'] = yaw_list

    proposal_df['x'] = xyzs[:,0]

    proposal_df['y'] = xyzs[:,1]

    proposal_df['z'] = xyzs[:,2]

    proposal_df['w'] = wlhs[:,0]

    proposal_df['l'] = wlhs[:,1]

    proposal_df['h'] = wlhs[:,2]

    

    proposal_df['corners'] = corners_list

    proposal_df['dimensions'] = dimensions_list

    proposal_df['candidate'] = equisized

    if(len(equisized)> 0):

        proposal_df['token'] = sample_token

    return all_points, has_rows, proposal_df
!pip install --upgrade 'git+git://github.com/wandb/client.git@object3D/lidar#egg=wandb' -qq
import wandb

from wandb.keras import WandbCallback

wandb.init(anonymous="allow")
# Render pointcloud, annotations and object proposals in 3D

def render_act_vs_pred_boxes_in_world(

        points,

        act_boxes,

        pred_box_corners

    ):



        c = np.array([255, 158, 0]) / 255.0

        

        df_tmp = pd.DataFrame(points, columns=["x", "y", "z"])

        df_tmp["norm"] = np.sqrt(np.power(df_tmp[["x", "y", "z"]].values, 2).sum(axis=1))

        '''

        cat = pd.cut(df_tmp["norm"], 10)

        print(cat)

        if( df_tmp["norm"] < 2368.356 ):

            df_tmp["cat"] = 0

        elif ( df_tmp["norm"] < 2368.356 ):

        (2368.356, 2385.992] < (2385.992, 2403.629] < (2403.629, 2421.265] ... (2456.538, 2474.174] < (2474.174, 2491.811] < (2491.811, 2509.447] < (2509.447, 2527.084]]

        # df_tmp["cat"]

        '''



        min_val = df_tmp["norm"].min()

        max_val = df_tmp["norm"].max()

        

        # print(df_tmp.describe())

        df_tmp["norm"] = (df_tmp["norm"]-min_val)/(max_val-min_val)

#         rgb = np.array([])

#         rgb = df_tmp["norm"]

#         g = (150*rgb)+50

#         b = (50*rgb)+100

        rgb = np.array([colors.hsv_to_rgb([n, 0.4, 0.5]) for n in df_tmp["norm"]]) * 255.0

        

        # print(df_tmp.describe())

        # print(rgb)

        

        points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(points, rgb)])

        # print("\n\nPoints RGB", points_rgb)

        

        

        scatter = go.Scatter3d(

            x=df_tmp["x"],

            y=df_tmp["y"],

            z=df_tmp["z"],

            mode="markers",

            marker=dict(size=1, color=df_tmp["norm"], opacity=0.8),

        )

        

        # print("Tmp: ", df_tmp)

        

        # wandb.log({"cluster": scatter})

        # act_boxes = np.array(act_boxes)

        

        # print("Describe: ", df_tmp.describe(),"\n\n")

        wandb.init(anonymous="allow")

        # wandb.log({"points": wandb.Object3D(np.array(points))})

        

        # Boxes:  label: nan, score: nan, xyz: [2172.19, 989.22, -18.46], wlh: [2.08, 5.25, 1.97], rot axis: [0.00, 0.00, 1.00], ang(degrees): -31.70, ang(rad): -0.55, vel: nan, nan, nan, name: car, token: 0504e4480bc4e9aad8c6bd40f1f2311d57379f978201338bbf369f37f1e7b6d2 





        # print("---------\n\nBoxes Shape: ", act_boxes.shape)

        boxes = []

        for i, box in enumerate(act_boxes):

            # print("XYZ: ", box.center[0],"\n--------------")

            # print("WLH: ", box.wlh[0],"\n--------------")

            corner_pts = box.corners()

            corner_pts_t = box.corners().transpose()

            # print("---------\n\nCorner Points Shape: ", corner_pts_t.shape)

            # print("---------\n\nCorner Points Transposed: ", corner_pts_t)

            boxes.append(corner_pts_t)

        print("---------\n\nBoxes: ", boxes)

        print("---------\n\nType(Boxes): ", type(boxes))

        wandb.log(

            {

                "point_scene": wandb.Object3D(

                    {

                        "type": "lidar/beta",

                        "points": np.array([[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]),

                        "boxes": np.array(boxes)

                    }

                )

            }

        )

        '''

        wandb.log(

        {

            "point_clouds_with_bb": wandb.Object3D(

                {

                    "type": "lidar/beta",

                    "points": points_rgb,

                    "boxes": boxes

                }

            )

        })

            a_dict = {

                        "x": float(box.center[0]),

                        "y": float(box.center[1]),

                        "z": float(box.center[2]),

                        "width": float(box.wlh[0]),

                        "length": float(box.wlh[1]),

                        "height": float(box.wlh[2]),

                        "axis": list(box.orientation.axis),

                        "rotation": float(box.orientation.radians)

                    }

            boxes.append(a_dict)

        

        boxes = np.array(boxes)

        

        print("Points:", points)

        print("\n\nBoxes: ", boxes,"\n\n")

        

        wandb.log(

        {

            "point_clouds_with_bb": wandb.Object3D(

                {

                    "type": "lidar/beta",

                    "points": points_rgb,

                    "boxes": boxes

                }

            )

        })

        '''

        

        x_lines = []

        y_lines = []

        z_lines = []



        def f_lines_add_nones():

            x_lines.append(None)

            y_lines.append(None)

            z_lines.append(None)



        ixs_box_0 = [0, 1, 2, 3, 0]

        ixs_box_1 = [4, 5, 6, 7, 4]



        for box in act_boxes:

            bpoints = view_points(box.corners(), view=np.eye(3), normalize=False)

            x_lines.extend(bpoints[0, ixs_box_0])

            y_lines.extend(bpoints[1, ixs_box_0])

            z_lines.extend(bpoints[2, ixs_box_0])

            f_lines_add_nones()

            x_lines.extend(bpoints[0, ixs_box_1])

            y_lines.extend(bpoints[1, ixs_box_1])

            z_lines.extend(bpoints[2, ixs_box_1])

            f_lines_add_nones()

            for i in range(4):

                x_lines.extend(bpoints[0, [ixs_box_0[i], ixs_box_1[i]]])

                y_lines.extend(bpoints[1, [ixs_box_0[i], ixs_box_1[i]]])

                z_lines.extend(bpoints[2, [ixs_box_0[i], ixs_box_1[i]]])

                f_lines_add_nones()



        lines = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name="lines")

        

        cx_lines = []

        cy_lines = []

        cz_lines = []



        def cf_lines_add_nones():

            cx_lines.append(None)

            cy_lines.append(None)

            cz_lines.append(None)



        cixs_box_0 = [0, 1, 2, 3, 0]

        cixs_box_1 = [4, 5, 6, 7, 4]





        for corners in pred_box_corners:

            cpoints = view_points(corners.T, view=np.eye(3), normalize=False)

            cx_lines.extend(cpoints[0, cixs_box_0])

            cy_lines.extend(cpoints[1, cixs_box_0])

            cz_lines.extend(cpoints[2, cixs_box_0])

            cf_lines_add_nones()

            cx_lines.extend(cpoints[0, cixs_box_1])

            cy_lines.extend(cpoints[1, cixs_box_1])

            cz_lines.extend(cpoints[2, cixs_box_1])

            cf_lines_add_nones()

            for i in range(4):

                cx_lines.extend(cpoints[0, [cixs_box_0[i], cixs_box_1[i]]])

                cy_lines.extend(cpoints[1, [cixs_box_0[i], cixs_box_1[i]]])

                cz_lines.extend(cpoints[2, [cixs_box_0[i], cixs_box_1[i]]])

                cf_lines_add_nones()



        clines = go.Scatter3d(x=cx_lines, y=cy_lines, z=cz_lines, mode="lines", name="clines")



        

        fig = go.Figure(data=[scatter,lines,clines])

        # fig = go.Figure(data=[scatter])

        fig.update_layout(scene_aspectmode="data")

        fig.show()

        

        print("---------\nDone")
# render_clustering(1,40)
def get_sample_tokens(log_token):

    scene_df =  pd.DataFrame(LYFT.scene)

    scene_df =  scene_df[scene_df['log_token']==log_token]

    scene_df.rename(columns={'token':'scene_token'}, inplace=True)



    sample_df = pd.DataFrame(LYFT.sample)



    s_df = pd.merge(sample_df, scene_df, left_on='scene_token', right_on='scene_token',how='inner')

    s_df = s_df.iloc[:10]

    return s_df['token']



# Compute rough accuracy and mean IOU for given hyperparameters

def score_clustering(eps, min_samples):

    log_token= '71dfb15d2f88bf2aab2c5d4800c0d10a76c279b9fda98720781a406cbacc583b'

    tokens = get_sample_tokens(log_token)

    tokens = tokens[:2]

    batch_input = extract_data_for_clustering(tokens)

    

    ious = []

    numcorrect = 0

    total = 0

    pred = 0

    for sample_token in tokens:

        token_input = batch_input[batch_input['sample_token'] == sample_token]

        pointcloud, has_rows, proposals_df = identify_clusters(sample_token, token_input, DATA_PATH, eps=eps, min_samples=min_samples)

        sampledata_token = token_input.iloc[0]['sampledata_token']

        pred = pred + len(proposals_df)

        boxes = LYFT.get_boxes(sampledata_token)

        

        for i  in range(len(boxes)):

            total = total+1

    

            matches = 0

            box = boxes[i]

            act_corners = box.corners().T

            for j in range(len(proposals_df)):

                pred_corners = np.array(proposals_df.iloc[j]['corners'])

                ioveru = iou(act_corners, pred_corners)

                if ioveru > 0:

                    matches = matches + 1

                    ious.append(ioveru)

            if (matches == 1):

                numcorrect = numcorrect + 1

            if (matches == 0):

                ious.append(0)

    print("eps:%.2f min_samples:%d -> correct:%d actual:%d  pred %d acc:%.4f mean-iou:%.4f max-iou:%.4f"% (eps, min_samples , numcorrect, total, pred, numcorrect / total, np.mean(ious), np.max(ious))  )

# Render points , actual and proposed bounding boxes in 3d 

def render_clustering(eps, min_samples):

    log_token= '71dfb15d2f88bf2aab2c5d4800c0d10a76c279b9fda98720781a406cbacc583b'

    tokens = get_sample_tokens(log_token)

    tokens = tokens[:2]

    batch_input = extract_data_for_clustering(tokens)

    sample_token = tokens[0]

    token_input = batch_input[batch_input['sample_token'] == sample_token]

    sampledata_token = token_input.iloc[0]['sampledata_token']

    pointcloud, has_rows, proposals_df = identify_clusters(sample_token, token_input, DATA_PATH, eps=eps, min_samples=min_samples)

    plt = render_act_vs_pred_boxes_in_world(pointcloud, LYFT.get_boxes(sampledata_token), proposals_df['corners'])

    print(pointcloud)
render_clustering(1,40)
# grid search

for eps in np.arange(1.7,1.8,0.5):

    for min_samples in np.arange(4, 6, 1):

         score_clustering(eps, min_samples)
score_clustering(1, 40)