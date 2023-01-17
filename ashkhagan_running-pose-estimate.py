!pip install -qq git+https://www.github.com/ildoonet/tf-pose-estimation
!pip install -qq pycocotools
%load_ext autoreload

%autoreload 2

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
%matplotlib inline

import tf_pose

import cv2

from glob import glob

from tqdm import tqdm_notebook

from PIL import Image

import numpy as np

import os

def video_gen(in_path):

    c_cap = cv2.VideoCapture(in_path)

    while c_cap.isOpened():

        ret, frame = c_cap.read()

        if not ret:

            break

        yield c_cap.get(cv2.CAP_PROP_POS_MSEC), frame[:, :, ::-1]

    c_cap.release()
video_paths = glob('../input/damian/*.mp4')

c_video = video_gen(video_paths[0])

for _ in range(300):

    c_ts, c_frame = next(c_video)

plt.imshow(c_frame)
from tf_pose.estimator import TfPoseEstimator

from tf_pose.networks import get_graph_path, model_wh

tfpe = tf_pose.get_estimator()
humans = tfpe.inference(npimg=c_frame, upsample_size=4.0)

print(humans)
new_image = TfPoseEstimator.draw_humans(c_frame[:, :, ::-1], humans, imgcopy=False)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

ax1.imshow(new_image[:, :, ::-1])
body_to_dict = lambda c_fig: {'bp_{}_{}'.format(k, vec_name): vec_val 

                              for k, part_vec in c_fig.body_parts.items() 

                              for vec_name, vec_val in zip(['x', 'y', 'score'],

                                                           (part_vec.x, 1-part_vec.y, part_vec.score))}

c_fig = humans[0]

body_to_dict(c_fig)
MAX_FRAMES = 3200

body_pose_list = []

for vid_path in tqdm_notebook(video_paths, desc='Files'):

    c_video = video_gen(vid_path)

    c_ts, c_frame = next(c_video)

    out_path = '{}_out.avi'.format(os.path.split(vid_path)[1])

    out = cv2.VideoWriter(out_path,

                          cv2.VideoWriter_fourcc('M','J','P','G'),

                          10, 

                          (c_frame.shape[1], c_frame.shape[0]))

    for (c_ts, c_frame), _ in zip(c_video, 

                                  tqdm_notebook(range(MAX_FRAMES), desc='Frames')):

        bgr_frame = c_frame[:,:,::-1]

        humans = tfpe.inference(npimg=bgr_frame, upsample_size=4.0)

        for c_body in humans:

            body_pose_list += [dict(video=out_path, time=c_ts, **body_to_dict(c_body))]

        new_image = TfPoseEstimator.draw_humans(bgr_frame, humans, imgcopy=False)

        out.write(new_image)

    out.release()

import pandas as pd

body_pose_df = pd.DataFrame(body_pose_list)

body_pose_df.describe()
fig, m_axs = plt.subplots(1, 2, figsize=(15, 5))

for c_ax, (c_name, c_rows) in zip(m_axs, body_pose_df.groupby('video')):

    for i in range(17):

        c_ax.plot(c_rows['time'], c_rows['bp_{}_y'.format(i)], label='x {}'.format(i))

    c_ax.legend()

    c_ax.set_title(c_name)
fig, m_axs = plt.subplots(1, 2, figsize=(15, 5))

for c_ax, (c_name, n_rows) in zip(m_axs, body_pose_df.groupby('video')):

    for i in range(17):

        c_rows = n_rows.query('bp_{}_score>0.6'.format(i)) # only keep confident results

        c_ax.plot(c_rows['bp_{}_x'.format(i)], c_rows['bp_{}_y'.format(i)], label='BP {}'.format(i))

    c_ax.legend()

    c_ax.set_title(c_name)
body_pose_df.to_csv('body_pose.csv', index=False)