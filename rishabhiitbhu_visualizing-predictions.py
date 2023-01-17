!pip install -U git+https://github.com/lyft/nuscenes-devkit
from copy import deepcopy

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go



from lyft_dataset_sdk.lyftdataset import LyftDataset, Box, Quaternion, view_points



%matplotlib inline
df = pd.read_csv('../input/lyftpredictionvisualization/lyft.csv')
df.head()
# some black magic, see https://www.kaggle.com/seshurajup/starter-lyft-level-5-av-dataset-from-github#625566

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data
lyft = LyftDataset(data_path='.', json_path='data', verbose=False)
def get2Box(boxes_list, names_list, token, scores=None):

    '''Given a list of boxes in `x,y,z,w,l,h,yaw` format, returns them in `Box` format

    

    Args:

    boxes_list: a list of boxes in [x, y, z, w, l, h, yaw] format

    names_list: classes the boxes belong to

    token: token of the sample the boxes belong to

    scores: predicted confidence scores, only for predicted boxes, 

    '''

    boxes = []

    for idx in range(len(boxes_list)):

        center = boxes_list[idx, :3] # x, y, z

        yaw = boxes_list[idx, 6]

        size = boxes_list[idx, 3:6] # w, l, h

        name = names_list[idx]

        detection_score = 1.0 # for ground truths 

        if scores is not None:

            detection_score = scores[idx]

        quat = Quaternion(axis=[0, 0, 1], radians=yaw)

        box = Box(

            center=center,

            size=size,

            orientation=quat,

            score=detection_score,

            name=name,

            token=token

        )

        boxes.append(box)

    return boxes

def get_pred_gt(pred_df, idx): 

    '''Given an index `idx`, this function reads ground truth and predicted strings and returns

    corresponding boxes in `Box` format'''

    

    sample_token = pred_df.iloc[idx]['Id']

    

    string = pred_df.iloc[idx]['GroundTruthString'].split()

    gt_objects = [string[x:x+8] for x in range(0, len(string), 8)]

    

    string = pred_df.iloc[idx]['PredictionString'].split()

    pred_objects = [string[x:x+9] for x in range(0, len(string), 9)]

    

    # str -> float, in x,y,z,w,l,h,yaw format

    gt_boxes = np.array([list(map(float, x[0:7])) for x in gt_objects])

    gt_class = np.array([x[7] for x in gt_objects])

    

    pred_scores = np.array([float(x[0]) for x in pred_objects])

    pred_boxes = np.array([list(map(float, x[1:8])) for x in pred_objects])

    pred_class = np.array([x[8] for x in pred_objects])

    

    # x,y,z,w,l,h,yaw -> Box instance

    predBoxes = get2Box(pred_boxes, pred_class, sample_token, scores=pred_scores)

    gtBoxes = get2Box(gt_boxes, gt_class, sample_token)

    

    return predBoxes, gtBoxes 
def glb_to_sensor(box, sample_data):

    '''Get a box from global frame to sensor's frame of reference '''

    

    box = box.copy() # v.imp

    cs_record = lyft.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    pose_record = lyft.get('ego_pose', sample_data['ego_pose_token'])

    

    # global to ego 

    box.translate(-np.array(pose_record['translation']))

    box.rotate(Quaternion(pose_record['rotation']).inverse)

    

    # ego to sensor

    box.translate(-np.array(cs_record['translation']))

    box.rotate(Quaternion(cs_record['rotation']).inverse)

    

    return box
# get predicted and ground boxes for each sample in `Box` format

pred_boxes = []

gt_boxes = []

for idx in range(len(df)):

    pBoxes, gBoxes = get_pred_gt(df, idx)

    pred_boxes.append(pBoxes)

    gt_boxes.append(gBoxes)
# let's take a peek

idx = 0

pred_boxes[idx][0]
gt_boxes[idx][0]
idx = 0 # change this to visualize other samples

sample_token = df.iloc[idx]['Id']



sample = lyft.get('sample', sample_token)

sample_data = lyft.get('sample_data', sample['data']['LIDAR_TOP'])

path = sample_data['filename']

lidar_points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]



_, ax = plt.subplots(1, 1, figsize=(9, 9))



# create colors based on the distance of the point from lidar

axes_limit=40



dists = np.sqrt(np.sum(lidar_points[:, :2] ** 2, axis=1))

colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c=colors, s=0.2)

ax.plot(0, 0, "x", color="red") # plot lidar location



# Limit visible range.

ax.set_xlim(-axes_limit, axes_limit)

ax.set_ylim(-axes_limit, axes_limit)



# plot the ground truths

for box in gt_boxes[idx]:

    box = glb_to_sensor(box, sample_data)

    c = np.array([255, 158, 0 ]) / 255.0 # Orange

    box.render(ax, view=np.eye(4), colors=(c, c, c))



# plot the predicted boxes

for box in pred_boxes[idx]:

    box = glb_to_sensor(box, sample_data)

    c = np.array([0, 0, 230]) / 255.0 # Blue

    box.render(ax, view=np.eye(4), colors=(c, c, c))



# gotta invert for consistency with lyft's inbuilt plots

plt.gca().invert_yaxis()

plt.gca().invert_xaxis()

plt.show()
lyft.render_sample_data(sample_data['token'])
def get_lines(boxes, name):

    '''Takes in boxes, extracts edges and returns `go.Scatter3d` object for those lines'''

    

    x_lines = []

    y_lines = []

    z_lines = []



    def f_lines_add_nones():

        x_lines.append(None)

        y_lines.append(None)

        z_lines.append(None)



    ixs_box_0 = [0, 1, 2, 3, 0]

    ixs_box_1 = [4, 5, 6, 7, 4]



    for box in boxes:

        box = glb_to_sensor(box, sample_data)

        points = view_points(box.corners(), view=np.eye(3), normalize=False)

        x_lines.extend(points[0, ixs_box_0])

        y_lines.extend(points[1, ixs_box_0])

        z_lines.extend(points[2, ixs_box_0])

        f_lines_add_nones()

        x_lines.extend(points[0, ixs_box_1])

        y_lines.extend(points[1, ixs_box_1])

        z_lines.extend(points[2, ixs_box_1])

        f_lines_add_nones()

        for i in range(4):

            x_lines.extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])

            y_lines.extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])

            z_lines.extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])

            f_lines_add_nones()



    lines = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name=name)

    return lines
idx = 0 # change this to visualize other samples

sample_token = df.iloc[idx]['Id']



sample = lyft.get("sample", sample_token)

sample_data = lyft.get("sample_data", sample["data"]["LIDAR_TOP"])

path = sample_data['filename']

lidar_points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]



# plot the points

df_tmp = pd.DataFrame(lidar_points[:, :3], columns=["x", "y", "z"])

df_tmp["norm"] = np.sqrt(np.power(df_tmp[["x", "y", "z"]].values, 2).sum(axis=1))

scatter = go.Scatter3d(

    x=df_tmp["x"],

    y=df_tmp["y"],

    z=df_tmp["z"],

    mode="markers",

    marker=dict(size=1, color=df_tmp["norm"], opacity=0.8),

)



gt_lines = get_lines(gt_boxes[idx], 'gt_boxes')

pred_lines = get_lines(pred_boxes[idx], 'pred_boxes')

fig = go.Figure(data=[scatter, gt_lines, pred_lines])

fig.update_layout(scene_aspectmode="data")

fig.show()