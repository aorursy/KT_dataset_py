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
import math
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
tfpe = tf_pose.get_estimator()
DEF_SHAPE = 256
im_side = cv2.imread('../input/pose-detection/unnamed.jpg')
res_side = cv2.resize(im_side, dsize=(DEF_SHAPE,DEF_SHAPE))
#print(im_side.shape)
# print(res_side.shape)
plt.imshow(res_side)
pts_side = tfpe.inference(npimg=res_side, upsample_size=4.0)
print(pts_side)
new_image = TfPoseEstimator.draw_humans(res_side, pts_side, imgcopy=False)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.imshow(new_image[:, :, ::-1])
body_to_dict = lambda c_fig: {'bp_{}_{}'.format(k, vec_name): vec_val 
                              for k, part_vec in c_fig.body_parts.items() 
                              for vec_name, vec_val in zip(['x', 'y', 'score'],
                                                           (part_vec.x, 1-part_vec.y, part_vec.score))}
c_fig_side = pts_side[0]
body_to_dict(c_fig_side)
# print(pts_side[0].body_parts.items())
count = 0
shldr=[0,0]
hip=[0,0]
knee=[0,0]
ear=[0,0]
for k, part_vec in pts_side[0].body_parts.items():
    if k==5:
        count+=1
        shldr[0]=part_vec.x
        shldr[1]=part_vec.y
    if k==11:
        count+=1
        hip[0]=part_vec.x
        hip[1]=part_vec.y
    if k==12:
        count+=1
        knee[0]=part_vec.x
        knee[1]=part_vec.y
    if k==17:
        count+=1
        ear[0]=part_vec.x
        ear[1]=part_vec.y
if count<4:
    print("image not ideal for work")
def length(arr1,arr2):
    return (math.sqrt((arr1[0]-arr2[0])**2+(arr1[1]-arr2[1])**2))
ang =math.acos((((ear[0]-shldr[0])*(shldr[0]-hip[0]))+((ear[1]-shldr[1])*(shldr[1]-hip[1])))/
               ((length(ear,shldr))*(length(shldr, hip))))
ang = math.degrees(ang)*100/90
print(ang,"%")
hipt = ((knee[0]-shldr[0])/(knee[1]-shldr[1]))*(hip[1]-knee[1])+knee[0]
slch_scr = abs((hipt-hip[0])/(knee[1]-shldr[1])*100)
print(hipt)
print(slch_scr,"%")
eart = ((hip[0]-shldr[0])/(hip[1]-shldr[1]))*(ear[1]-shldr[1])+shldr[0]
kypho_scr = abs((eart-ear[0])/(hip[1]-shldr[1])*100)
print(eart)
print(kypho_scr,"%")
hipt = ((ear[0]-knee[0])/(ear[1]-knee[1]))*(hip[1]-ear[1])+ear[0]
lordo_scr = abs((hipt-hip[0])/(ear[1]-knee[1]))*100
print(hipt)
print(lordo_scr,"%")
im_front = cv2.imread('../input/pose-detection/Josh-Simson.jpg')
res_front = cv2.resize(im_front, dsize=(DEF_SHAPE,DEF_SHAPE))
plt.imshow(res_front)
pts_front = tfpe.inference(npimg=res_front, upsample_size=4.0)
print(pts_front)
new_image = TfPoseEstimator.draw_humans(res_front, pts_front, imgcopy=False)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.imshow(new_image[:, :, ::-1])
body_to_dict = lambda c_fig: {'bp_{}_{}'.format(k, vec_name): vec_val 
                              for k, part_vec in c_fig.body_parts.items() 
                              for vec_name, vec_val in zip(['x', 'y', 'score'],
                                                           (part_vec.x, 1-part_vec.y, part_vec.score))}
c_fig_front = pts_front[0]
body_to_dict(c_fig_front)
count = 0
left_hip=[0,0]
right_hip=[0,0]
left_knee=[0,0]
right_knee=[0,0]
left_foot=[0,0]
right_foot=[0,0]
for k, part_vec in pts_front[0].body_parts.items():
    if k==8:
        count+=1
        right_hip[0]=part_vec.x
        right_hip[1]=part_vec.y
    if k==9:
        count+=1
        right_knee[0]=part_vec.x
        right_knee[1]=part_vec.y
    if k==10:
        count+=1
        right_foot[0]=part_vec.x
        right_foot[1]=part_vec.y
    if k==11:
        count+=1
        left_hip[0]=part_vec.x
        left_hip[1]=part_vec.y
    if k==12:
        count+=1
        left_knee[0]=part_vec.x
        left_knee[1]=part_vec.y
    if k==13:
        count+=1
        left_foot[0]=part_vec.x
        left_foot[1]=part_vec.y
if count<6:
    print("image not ideal to work")
l_knee_t = ((left_foot[0]-left_hip[0])/(left_foot[1]-left_hip[1]))*(left_knee[1]-left_foot[1])+left_foot[0]
left_leg_scr = (-1)*((l_knee_t-left_knee[0])/(left_foot[1]-left_hip[1]))*250
print(l_knee_t)
print(left_leg_scr,"%")
r_knee_t = ((right_foot[0]-right_hip[0])/(right_foot[1]-right_hip[1]))*(right_knee[1]-right_foot[1])+right_foot[0]
right_leg_scr = ((r_knee_t-right_knee[0])/(right_foot[1]-right_hip[1]))*250
print(r_knee_t)
print(right_leg_scr,"%")
