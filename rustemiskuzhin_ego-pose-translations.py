# Operating system

import sys

import os

from pathlib import Path



# math

import numpy as np



# data analysis

import pandas as pd



#plotting 2D

import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from matplotlib import animation, rc

import matplotlib.patches as patches

import matplotlib as mpl
# Lyft dataset SDK

!pip install lyft-dataset-sdk

from lyft_dataset_sdk.lyftdataset import LyftDataset

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion

from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar
lyft_dataset =  LyftDataset(data_path='.', json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)

log_df = pd.DataFrame(lyft_dataset.log)

# log_df = log_df[log_df['vehicle'].str.match('a101')]

#da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e

scene_df =  pd.DataFrame(lyft_dataset.scene)

scene_df = pd.merge(log_df, scene_df, left_on='token', right_on='log_token',how='inner')



# scene_df.head()

sample_df = pd.DataFrame(lyft_dataset.sample)

sample_df = pd.merge(scene_df[['log_token', 'date_captured', 'vehicle', 'token_y']], sample_df, left_on='token_y', right_on='scene_token',how='inner')

# sample_df.head()



sampledata_df = pd.DataFrame(lyft_dataset.sample_data)

sampledata_df = pd.merge(sample_df[['log_token', 'date_captured', 'token', 'vehicle']], sampledata_df, left_on='token', right_on='sample_token',how='inner')

# sampledata_df.head()

counts = sampledata_df.groupby(['vehicle','date_captured'])['channel'].value_counts().unstack().fillna(0)



counts
# join log, scene, sample, data, ego pose and filter for car a102's ride on 2019-05-24

log_df = pd.DataFrame(lyft_dataset.log)

log_df = log_df[log_df['date_captured'].str.match('2019-05-24')]

log_df = log_df[log_df['vehicle'].str.match('a102')]





scene_df =  pd.DataFrame(lyft_dataset.scene)

scene_df = pd.merge(log_df, scene_df, left_on='token', right_on='log_token',how='inner')



sample_df = pd.DataFrame(lyft_dataset.sample)

sample_df = pd.merge(sample_df, scene_df[['vehicle', 'token_y']], left_on='scene_token', right_on='token_y',how='inner')



sampledata_df = pd.DataFrame(lyft_dataset.sample_data)

sampledata_df = pd.merge(sample_df[['token', 'vehicle']], sampledata_df, left_on='token', right_on='sample_token',how='inner')



ego_pose_df = pd.DataFrame(lyft_dataset.ego_pose)

ego_pose_df = pd.merge(sampledata_df[['token_x','ego_pose_token', 'channel','vehicle' ]], 

                                   ego_pose_df, left_on='ego_pose_token', right_on='token',how='inner')



ego_pose_df = ego_pose_df.drop(['token'], axis=1)

ego_pose_df.rename(columns={'token_x':'token'}, inplace=True)



# ego_pose_df = ego_pose_df[ego_pose_df['vehicle'].str.match('a101')]

ego_pose_df.sort_values(by=['timestamp'])



calibrated_sensor_df = pd.DataFrame(lyft_dataset.calibrated_sensor)



# pivot on sample token to spread channel translations across columns

pivot_df = ego_pose_df.pivot(index ='token', columns ='channel', values = ['translation','rotation']).reset_index()

ego_pose_df.head()
calibrated_sensor_df.head()
len(calibrated_sensor_df)
pivot_df.head()
center_x = []

center_y = []

x = []

y = []

x0 = []

y0 = []

x1 = []

y1 = []

x2 = []

y2 = []

x3 = []

y3 = []

x4 = []

y4 = []

x5 = []

y5 = []

x6 = []

y6 = []

x7 = []

y7 = []

yaw = []

num_sample = len(pivot_df)

for i in range(num_sample):

    token = pivot_df.iloc[i, 0]

    my_sample = lyft_dataset.get('sample', token)

    sample_lidar_token = my_sample["data"]['LIDAR_TOP']

    cam = lyft_dataset.get("sample_data", sample_lidar_token)

    poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    yaw.append(Quaternion(poserecord["rotation"]).yaw_pitch_roll[0])

    

    center_x.append(poserecord["translation"][0])

    center_y.append(poserecord["translation"][1])

    

    sample_cam_token = my_sample["data"]['CAM_FRONT']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    

    x.append(poserecord["translation"][0] + sensor_vector[0])

    y.append(poserecord["translation"][1] + sensor_vector[1])

    

    sample_cam_token = my_sample["data"]['CAM_FRONT_LEFT']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x0.append(poserecord["translation"][0] + sensor_vector[0])

    y0.append(poserecord["translation"][1] + sensor_vector[1])



    sample_cam_token = my_sample["data"]['CAM_FRONT_RIGHT']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x1.append(poserecord["translation"][0] + sensor_vector[0])

    y1.append(poserecord["translation"][1] + sensor_vector[1])



    sample_lidar_token = my_sample["data"]['LIDAR_TOP']

    cam = lyft_dataset.get("sample_data", sample_lidar_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x2.append(poserecord["translation"][0] + sensor_vector[0])

    y2.append(poserecord["translation"][1] + sensor_vector[1])



    sample_cam_token = my_sample["data"]['CAM_BACK']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x3.append(poserecord["translation"][0] + sensor_vector[0])

    y3.append(poserecord["translation"][1] + sensor_vector[1])

    

    sample_cam_token = my_sample["data"]['CAM_BACK_LEFT']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x4.append(poserecord["translation"][0] + sensor_vector[0])

    y4.append(poserecord["translation"][1] + sensor_vector[1])

    

    sample_cam_token = my_sample["data"]['CAM_BACK_RIGHT']

    cam = lyft_dataset.get("sample_data", sample_cam_token)

    #poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x5.append(poserecord["translation"][0] + sensor_vector[0])

    y5.append(poserecord["translation"][1] + sensor_vector[1])

    

    sample_token = my_sample["data"]['LIDAR_FRONT_LEFT']

    cam = lyft_dataset.get("sample_data", sample_token)

    poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x6.append(poserecord["translation"][0] - sensor_vector[0])

    y6.append(poserecord["translation"][1] - sensor_vector[1])

    

    sample_token = my_sample["data"]['LIDAR_FRONT_RIGHT']

    cam = lyft_dataset.get("sample_data", sample_token)

    poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

    cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])

    sensor_vector = np.dot(Quaternion(poserecord["rotation"]).rotation_matrix, cs_record["translation"])

    

    x7.append(poserecord["translation"][0] - sensor_vector[0])

    y7.append(poserecord["translation"][1] - sensor_vector[1])
cx = sorted(center_x)

cx
fig = plt.figure(figsize=(16,8))

        

ax = fig.add_subplot(111)

ax.set(xlim=(2110, 2130), ylim=(1020, 1035))



for i in range(num_sample):

    if i % 5 == 0:

        cx, cy = center_x[i], center_y[i]

        angle = yaw[i] * 57 #get angle in degrees

        rect = patches.Rectangle((-2.2,-1),4.4,2,linewidth=1,edgecolor='r',facecolor='none')

        t1 = mpl.transforms.Affine2D().rotate_deg(angle) + mpl.transforms.Affine2D().translate(cx,cy)

        rect.set_transform(t1 + ax.transData)

        ax.add_patch(rect)



ax.scatter(center_x, center_y, s=100, c='purple', label='Center')

ax.scatter(x, y,s=50, c='r', label='Camera Front')

ax.scatter(x0, y0,s=50, c='g', label='Camera Front Left')

ax.scatter(x1, y1,s=50, c='orange', label='Camera Front Right')

ax.scatter(x2, y2,s=50, c='b', label='LIDAR Top')

ax.scatter(x3, y3,s=50, c='y', label='Camera Back')

ax.scatter(x4, y4,s=50, c='k', label='Camera Back Left')

ax.scatter(x5, y5,s=50, c='m', label='Camera Back Right')

ax.scatter(x6, y6,s=50, c='c', label='LIDAR Front Left')

ax.scatter(x7, y7,s=50, c='brown', label='LIDAR Front Right')

ax.legend()

plt.show()
fig = plt.figure(figsize=(16,8))

        

ax = fig.add_subplot(111)

ax.set(xlim=(2000, 2400), ylim=(500, 1200))



ax.scatter(x, y,s=50, c='r', label='Camera Front')

ax.scatter(x0, y0,s=50, c='g', label='Camera Front Left')

ax.scatter(x1, y1,s=50, c='orange', label='Camera Front Right')

ax.scatter(x2, y2,s=50, c='b', label='LIDAR Top')

ax.scatter(x3, y3,s=50, c='y', label='Camera Back')

ax.scatter(x4, y4,s=50, c='k', label='Camera Back Left')

ax.scatter(x5, y5,s=50, c='m', label='Camera Back Right')

ax.scatter(x6, y6,s=50, c='c', label='LIDAR Front Left')

ax.scatter(x7, y7,s=50, c='brown', label='LIDAR Front Right')

ax.legend()

plt.show()