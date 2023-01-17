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



ego_pose_df = pd.merge(sampledata_df[['token_x','ego_pose_token', 'channel','vehicle' ,'calibrated_sensor_token']], 

                                   ego_pose_df, left_on='ego_pose_token', right_on='token',how='inner')



ego_pose_df = ego_pose_df.drop(['token'], axis=1)

ego_pose_df.rename(columns={'token_x':'token'}, inplace=True)





calibrated_sensor_df = pd.DataFrame(lyft_dataset.calibrated_sensor)

# calibrated_sensor_df.head()

calibrated_sensor_df.rename(columns={

    'token':'calibrated_sensor_token',

    'rotation':'calibrated_sensor_rotation',

    'translation':'calibrated_sensor_translation'    

                                    }, inplace=True)







ego_pose_df = pd.merge(ego_pose_df, 

                      calibrated_sensor_df[['calibrated_sensor_token', 'calibrated_sensor_rotation', 'calibrated_sensor_translation']],

                      left_on='calibrated_sensor_token',

                      right_on='calibrated_sensor_token',

                      how='inner')

# ego_pose_df = ego_pose_df[ego_pose_df['vehicle'].str.match('a101')]

# ego_pose_df.sort_values(by=['token','timestamp'])

ego_pose_df.sort_values(by=['token'])

ego_pose_df['timestamp'] = ego_pose_df['timestamp'].astype(str)



# pivot on sample token to spread channel translations across columns

pivot_df = ego_pose_df.pivot(index ='token', columns ='channel', values = ['translation','rotation','calibrated_sensor_translation','calibrated_sensor_rotation']).reset_index()

ego_pose_df[ego_pose_df['token'].str.match('a1e8c14fe99d3543b54adfefafc8207cb1c34a80afde92')]
pivot_df.head()


# Camera front

x = pivot_df.iloc[:,4].map(lambda t: t[0])

y = pivot_df.iloc[:,4].map(lambda t: t[1])





# Camera front left

x0 = pivot_df.iloc[:,5].map(lambda t: t[0])

y0 = pivot_df.iloc[:,5].map(lambda t: t[1])



# Camera front right

x1 = pivot_df.iloc[:,6].map(lambda t: t[0])

y1 = pivot_df.iloc[:,6].map(lambda t: t[1])





#LIDAR top

x2 = pivot_df.iloc[:,10].map(lambda t: t[0])

y2 = pivot_df.iloc[:,10].map(lambda t: t[1])





#Camera Back

x3 = pivot_df.iloc[:,1].map(lambda t: t[0])

y3 = pivot_df.iloc[:,1].map(lambda t: t[1])





fig = plt.figure(figsize=(16,8))

        

ax = fig.add_subplot(111)

ax.set(xlim=(2110, 2130), ylim=(1020, 1035))



ax.scatter(x, y,s=50, c='r', label='Camera Front')

ax.scatter(x0, y0,s=50, c='g', label='Camera Front Left')

ax.scatter(x1, y1,s=50, c='orange', label='Camera Front Right')

ax.scatter(x2, y2,s=50, c='b', label='LIDAR Top')

ax.scatter(x3, y3,s=50, c='y', label='Camera Back')



ax.legend()

plt.show()
# Camera front

real_camera_front_coords =  pivot_df.apply(lambda row: Quaternion(row.iloc[14]).rotate(row.iloc[4]), axis=1)

x_rot = real_camera_front_coords.map(lambda t: t[0])

y_rot = real_camera_front_coords.map(lambda t: t[1])





# Camera front left

real_camera_front_coords_0 =  pivot_df.apply(lambda row: Quaternion(row.iloc[15]).rotate(row.iloc[5]), axis=1)

x0_rot = real_camera_front_coords_0.map(lambda t: t[0])

y0_rot = real_camera_front_coords_0.map(lambda t: t[1])





# Camera front right

real_camera_front_coords_1 =  pivot_df.apply(lambda row: Quaternion(row.iloc[16]).rotate(row.iloc[6]), axis=1)

x1_rot = real_camera_front_coords_1.map(lambda t: t[0])

y1_rot = real_camera_front_coords_1.map(lambda t: t[1])







#LIDAR top

real_camera_front_coords_2 =  pivot_df.apply(lambda row: Quaternion(row.iloc[20]).rotate(row.iloc[10]), axis=1)

x2_rot = real_camera_front_coords_2.map(lambda t: t[0])

y2_rot = real_camera_front_coords_2.map(lambda t: t[1])







#Camera Back

real_camera_front_coords_3 =  pivot_df.apply(lambda row: Quaternion(row.iloc[11]).rotate(row.iloc[1]), axis=1)

x3_rot = real_camera_front_coords_3.map(lambda t: t[0])

y3_rot = real_camera_front_coords_3.map(lambda t: t[1])





row = pivot_df.iloc[0,[1,11,21,31]]

row
row = pivot_df.iloc[0,[4,14,24,34]]

row

#row.iloc[0], row.iloc[1],Quaternion(row.iloc[1]).rotate(row.iloc[0])
pivot_df.iloc[0,[5,15,25,35]]
pivot_df.iloc[0,[6,16,26,36]]
pivot_df.iloc[0,[1,11,21,31]]
pivot_df.iloc[0,[10,20,30,40]]
row = pivot_df.iloc[0,[5,15]]

row.iloc[0], row.iloc[1],Quaternion(row.iloc[1]).rotate(row.iloc[0])
row = pivot_df.iloc[0,[6,16]]

row.iloc[0], row.iloc[1],Quaternion(row.iloc[1]).rotate(row.iloc[0])
row = pivot_df.iloc[0,[1,11]]

row.iloc[0], row.iloc[1],Quaternion(row.iloc[1]).rotate(row.iloc[0])
pivot_df.iloc[0,[10,20]]
row = pivot_df.iloc[0,[10,20]]

row.iloc[0], row.iloc[1],Quaternion(row.iloc[1]).rotate(row.iloc[0])
fig = plt.figure(figsize=(16,8))

        



ax = fig.add_subplot(111)

ax.set(xlim=(2610, 2630), ylim=(-1270, -1225))



ax.scatter(x_rot, y_rot,s=50, c='r', label='Camera Front')

ax.scatter(x0_rot, y0_rot,s=50, c='g', label='Camera Front Left')

ax.scatter(x1_rot, y1_rot,s=50, c='orange', label='Camera Front Right')

ax.scatter(x2_rot, y2_rot,s=50, c='b', label='LIDAR Top')

ax.scatter(x3_rot, y3_rot,s=50, c='y', label='Camera Back')



ax.legend()



plt.show()
## Rotate transform sensor's coordinates in car frame (using the rotation specs of back camera as it is closest to back frame)

## Then position transform the latter in world frame

def sensor_coords_in_world (row, i):

    rot = row.iloc[11];

    return  np.add(row.iloc[1],Quaternion(rot).rotate(row.iloc[i+20] ))

pivot_df.iloc[0].iloc[10]
pivot_df.iloc[0].iloc[1],pivot_df.iloc[0].iloc[21], sensor_coords_in_world(pivot_df.iloc[0],1)

pivot_df.iloc[0].iloc[4],pivot_df.iloc[0].iloc[24], sensor_coords_in_world(pivot_df.iloc[0],4)
pivot_df.iloc[0].iloc[5],pivot_df.iloc[0].iloc[25], sensor_coords_in_world(pivot_df.iloc[0],5)
pivot_df.iloc[0].iloc[6], pivot_df.iloc[0].iloc[26], sensor_coords_in_world(pivot_df.iloc[0],6)
pivot_df.iloc[0].iloc[10], pivot_df.iloc[0].iloc[30], sensor_coords_in_world(pivot_df.iloc[0],10)




# Camera front

real_camera_front_coords =  pivot_df.apply(lambda row: sensor_coords_in_world(row,4), axis=1)

x_rot = real_camera_front_coords.map(lambda t: t[0])

y_rot = real_camera_front_coords.map(lambda t: t[1])





# Camera front left

real_camera_front_coords_0 =  pivot_df.apply(lambda row: sensor_coords_in_world(row,5), axis=1)

x0_rot = real_camera_front_coords_0.map(lambda t: t[0])

y0_rot = real_camera_front_coords_0.map(lambda t: t[1])





# Camera front right

real_camera_front_coords_1 =  pivot_df.apply(lambda row: sensor_coords_in_world(row,6), axis=1)

x1_rot = real_camera_front_coords_1.map(lambda t: t[0])

y1_rot = real_camera_front_coords_1.map(lambda t: t[1])







#LIDAR top

real_camera_front_coords_2 =  pivot_df.apply(lambda row: sensor_coords_in_world(row,10), axis=1)

x2_rot = real_camera_front_coords_2.map(lambda t: t[0])

y2_rot = real_camera_front_coords_2.map(lambda t: t[1])







#Camera Back

real_camera_front_coords_3 =  pivot_df.apply(lambda row: sensor_coords_in_world(row,1), axis=1)

x3_rot = real_camera_front_coords_3.map(lambda t: t[0])

y3_rot = real_camera_front_coords_3.map(lambda t: t[1])







fig = plt.figure(figsize=(16,8))

        



ax = fig.add_subplot(111)

ax.set(xlim=(2120, 2145), ylim=(1010, 1027))



ax.scatter(x_rot, y_rot,s=50, c='r', label='Camera Front')

ax.scatter(x0_rot, y0_rot,s=50, c='g', label='Camera Front Left')

ax.scatter(x1_rot, y1_rot,s=50, c='orange', label='Camera Front Right')

ax.scatter(x2_rot, y2_rot,s=50, c='b', label='LIDAR Top')

ax.scatter(x3_rot, y3_rot,s=50, c='y', label='Camera Back')



ax.legend()

plt.show()