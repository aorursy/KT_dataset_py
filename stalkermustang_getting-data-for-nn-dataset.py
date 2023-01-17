!pip install -qqq lyft-dataset-sdk
import os

import math



import cv2

import numpy as np

import pandas as pd

from pathlib import Path



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib as mpl



from lyft_dataset_sdk.lyftdataset import LyftDataset

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
INP_DIR = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'
# Load the dataset

# Adjust the dataroot parameter below to point to your local dataset path.

# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train

!ln -s {INP_DIR}/train_images images

!ln -s {INP_DIR}/train_maps maps

!ln -s {INP_DIR}/train_lidar lidar
level5data = LyftDataset(

    data_path='.',

    json_path=os.path.join(INP_DIR + 'train_data'),

    verbose=False

)
my_scene = level5data.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = level5data.get('sample', my_sample_token)
lidar_top_data_token = my_sample['data']['LIDAR_TOP']
lidar_top_data = level5data.get('sample_data', lidar_top_data_token)
# lidar_top_ego_pose_token = lidar_top_data['ego_pose_token']
# # car / lidar top coords

# lidar_top_ego_pose_data = level5data.get('ego_pose', lidar_top_ego_pose_token)

# lidar_top_coords = np.array(lidar_top_ego_pose_data['translation'])
def get_coords_from_ann_idx(ann_idx):

    return np.array(level5data.get('sample_annotation', my_sample['anns'][ann_idx])['translation'])
anns_inds_to_show = [13, 27, 41, 42, 56] # i select several cars near lyft's pod

ann_tokens = []

for ind in anns_inds_to_show:

    my_annotation_token = my_sample['anns'][ind]

    print(f'{ind}: {my_annotation_token}')

    ann_tokens.append(my_annotation_token)

    level5data.render_annotation(my_annotation_token)

    plt.show()
my_sample['data'].keys() # available sensors for sampling
# here we get all objects (bboxes) for selected sensor and filter by selected cars (above)

ret_sampled = level5data.get_sample_data(lidar_top_data_token, selected_anntokens=ann_tokens)[1]

# we can not pass selected_anntokens and get full info of the bboxes around 

# car - literally a complete set of data for training a neural network
def get_data_from_sample(chanel_to_get):

    return level5data.get('sample_data', my_sample['data'][chanel_to_get])
def show_img_from_data(data):

    plt.imshow(

        cv2.cvtColor(

            cv2.imread(data['filename']),

            cv2.COLOR_BGR2RGB

        )

    );
lidar_channel = 'LIDAR_TOP'

camera1_chanel = 'CAM_BACK'

camera2_chanel = 'CAM_BACK_LEFT'

camera3_chanel = 'CAM_FRONT'

lidar_data = get_data_from_sample(lidar_channel)

camera1_data = get_data_from_sample(camera1_chanel)

camera2_data = get_data_from_sample(camera2_chanel)

camera3_data = get_data_from_sample(camera3_chanel)
# car 1 on back side

show_img_from_data(camera1_data);
# car 2 on back side

show_img_from_data(camera2_data);
# car 3 on front side

show_img_from_data(camera3_data);
pc = LidarPointCloud.from_file(Path(lidar_data['filename']))
plot_offset = 15

plt.xlim(-plot_offset, plot_offset)

plt.ylim(-plot_offset, plot_offset)



# plot PointCloud

plt.scatter(pc.points[0, :], pc.points[1, :], c=pc.points[2, :], s=0.1);

# plot center of cars / bboxes

for cur_point_idx in range(len(ret_sampled)):

    crds = ret_sampled[cur_point_idx].center

    plt.scatter(crds[0], crds[1], c='red');
pc.points[:, 0]
(pc.points[3, :]==100).all()
sample_to_explore = ret_sampled[4]
sample_to_explore
# rotation in radians of bbox: x, y and z. For degrees see example above (next MPL Figure)

sample_to_explore.orientation.yaw_pitch_roll
# Width, leght and height of bbox: x, y and z

sample_to_explore.wlh
# and of course coords of bbox 

sample_to_explore.center
# class of object

sample_to_explore.name
fig = plt.figure()

ax = fig.add_subplot(111)



plot_offset = 15

plt.xlim(-plot_offset, plot_offset)

plt.ylim(-plot_offset, plot_offset)



# plot PointCloud

ax.scatter(pc.points[0, :], pc.points[1, :], c=pc.points[2, :], s=0.1);



# plot center of cars / bboxes

crds = sample_to_explore.center

ax.scatter(crds[0], crds[1], c='red');



w, l, h  = sample_to_explore.wlh

angles_to_rotate = sample_to_explore.orientation.yaw_pitch_roll



rect_angle = math.degrees(angles_to_rotate[0]) # radians to degrees

mpl_rotate = (

    mpl

    .transforms

    .Affine2D()

    .rotate_deg_around(crds[0], crds[1], rect_angle)

    + ax.transData

)



rect = patches.Rectangle(

    (crds[0] - l / 2, crds[1] - w / 2), l, w, fill=False,

)



rect.set_transform(mpl_rotate)



ax.add_patch(rect);