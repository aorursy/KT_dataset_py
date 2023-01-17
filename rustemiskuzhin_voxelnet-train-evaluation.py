!pip install lyft-dataset-sdk
!pip install squaternion
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt

import matplotlib.patches as patches

import pandas as pd

import numpy as np

# Load the SDK

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer

from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud, RadarPointCloud 

from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points

from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision, get_class_names, get_average_precisions

import math

# from source.utilities import print_progress

from pyquaternion import Quaternion

import os

import time

from multiprocessing import Process

from squaternion import euler2quat, quat2euler

from tqdm import tqdm_notebook as tqdm

from shapely.geometry import Point

from shapely.geometry.polygon import Polygon
# gotta do this for LyftDataset SDK, it expects folders to be named as `images`, `maps`, `lidar`



!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data
level5data = LyftDataset(data_path='.', json_path='data/', verbose=True)
train = pd.read_csv('../input/lyft3d-inference-kernel-train-dataset/lyft3d_pred_train.csv')

train_num = len(train)

print(train_num)
train.head()
pred_animal = pd.read_csv('../input/animal-evaluation/lyft3d_train_pred_animal.csv')

pred_animal.head()
pred_motorcycle = pd.read_csv('../input/motorcycle-train-evaluation/lyft3d_train_pred_motorcycle.csv')

pred_motorcycle.head()
def get_boxes_list_3d(pred_boxes):

    if not pd.isna(pred_boxes):

        pred_boxes = pred_boxes.split(' ')

        pred_boxes = pred_boxes[:-1]

        pred_boxes_list = np.reshape(pred_boxes, (len(pred_boxes)//9, 9))

    else:

        pred_boxes_list = []

    return pred_boxes_list
sample_tokens = []

for idx in range(train_num):

    sample_tokens.append(pred_animal.iloc[idx]['Id'])
columns = ['Id', 'PredictionString']

df_pred = pd.DataFrame(columns=columns)   
for i, sample_token in tqdm(enumerate(sample_tokens)):

    row = [] 

    result = ''

    token = sample_token

    row.append(token)

    df = train.loc[train['Id'] == token]

    pred_str = str(np.array(df['PredictionString'])[0])

    pred_str_animal = pred_animal.iloc[i]['PredictionString']

    df = pred_motorcycle.loc[pred_motorcycle['Id'] == token]

    pred_str_motorcycle = str(np.array(df['PredictionString'])[0])

    if pd.isna(pred_str_animal):

        pred_str_animal = ''

    if pd.isna(pred_str_motorcycle):

        pred_str_motorcycle = ''   

    result = pred_str + pred_str_animal + pred_str_motorcycle

    row.append(result)  

    df_row = pd.DataFrame([row], columns=columns)

    df_pred = df_pred.append(df_row)
df_pred.head()
df_pred.tail()
df_pred.to_csv('train_correct.csv',index=False)
def load_groundtruth_boxes(nuscenes, sample_tokens):

    gt_box3ds = []



    # Load annotations and filter predictions and annotations.

    for sample_token in tqdm(sample_tokens):



        sample = nuscenes.get('sample', sample_token)

        sample_annotation_tokens = sample['anns']

        

        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)

            sample_annotation_translation = sample_annotation['translation']

            

            class_name = sample_annotation['category_name']

            

            box3d = Box3D(

                sample_token=sample_token,

                translation=sample_annotation_translation,

                size=sample_annotation['size'],

                rotation=sample_annotation['rotation'],

                name=class_name

            )

            gt_box3ds.append(box3d)

            

    return gt_box3ds



gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)
len(gt_box3ds)
pred_box3ds = []

pred_box3ds_correct = []

for i, sample_token in tqdm(enumerate(sample_tokens)):

    row = [] 

    token = sample_token

    df = train.loc[train['Id'] == token]

    pred_boxes = str(np.array(df['PredictionString'])[0])

    pred_boxes_list_3d = get_boxes_list_3d(pred_boxes)

    for j, pred_box in enumerate(pred_boxes_list_3d):

        cls_conf, x, y, z, w, l, h, yaw, cls_pred = pred_box

        cls_conf = float(cls_conf)

        x = float(x)

        y = float(y)

        z = float(z)

        w = float(w)

        h = float(h)

        l = float(l)

        yaw = float(yaw)

        q = euler2quat(0, 0, yaw)

        box3d = Box3D(

                    sample_token=token,

                    translation=[x, y, z],

                    size=[w, l, h],

                    rotation=[q[0],q[1],q[2],q[3]],

                    name=cls_pred,

                    score=cls_conf

                    )

        pred_box3ds.append(box3d)

    pred_boxes_correct = df_pred.iloc[i]['PredictionString']

    pred_boxes_list_3d_correct = get_boxes_list_3d(pred_boxes_correct)  

    for j, pred_box in enumerate(pred_boxes_list_3d_correct):

        cls_conf, x, y, z, w, l, h, yaw, cls_pred = pred_box

        cls_conf = float(cls_conf)

        x = float(x)

        y = float(y)

        z = float(z)

        w = float(w)

        h = float(h)

        l = float(l)

        yaw = float(yaw)

        q = euler2quat(0, 0, yaw)

        box3d_correct = Box3D(

                    sample_token=token,

                    translation=[x, y, z],

                    size=[w, l, h],

                    rotation=[q[0],q[1],q[2],q[3]],

                    name=cls_pred,

                    score=cls_conf

                    )

        pred_box3ds_correct.append(box3d_correct)
print(len(pred_box3ds), len(pred_box3ds_correct))
print(pred_box3ds[0], pred_box3ds_correct[0])
import json

ARTIFACTS_FOLDER = "./artifacts"

os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

gt = [b.serialize() for b in gt_box3ds[:10000]]

pr = [b.serialize() for b in pred_box3ds[:10000]]
gt[0], pr[0]
iou_th_range = np.linspace(0.5, 0.95, 10)

metric = {}

processes = []

output_dir = 'tmp/'

output_dir = Path(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)
class_names = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle',

                    'motorcycle', 'other_vehicle', 'pedestrian', 'truck']
def save_AP(gt, predictions, class_names, iou_threshold, output_dir):

    #computes average precisions (AP) for a given threshold, and saves the metrics in a temp file 

    # use lyft's provided function to compute AP

    AP = get_average_precisions(gt, predictions, class_names, iou_threshold)

    # create a dict with keys as class names and values as their respective APs

    metric = {c:AP[idx] for idx, c in enumerate(class_names)}



    # save the dict in a temp file

    summary_path = str(output_dir) + f'metric_summary_{iou_threshold}.json'

    with open(str(summary_path), 'w') as f:

        json.dump(metric, f)
def get_metric_overall_AP(iou_th_range, output_dir, class_names):

    ''' reads temp files and calculates overall per class APs.

    returns:

        `metric`: a dict with key as iou thresholds and value as dicts of class and their respective APs,

        `overall_AP`: overall AP of each class

    '''



    metric = {}

    overall_AP = np.zeros(len(class_names))

    for iou_threshold in iou_th_range:

        summary_path = str(output_dir) + f'metric_summary_{iou_threshold}.json'

        with open(str(summary_path), 'r') as f:

            data = json.load(f) # type(data): dict

            metric[iou_threshold] = data

            overall_AP += np.array([data[c] for c in class_names])

    overall_AP /= len(iou_th_range)

    return metric, overall_AP
time_start = time.time()

for iou_threshold in iou_th_range:

    process = Process(target=save_AP, args=(gt, pr, class_names, iou_threshold, output_dir))

    process.start()

    processes.append(process)



for process in processes:

    process.join()

print("Time to evaluate = ", time.time() - time_start)      
# get overall metrics

metric, overall_AP = get_metric_overall_AP(iou_th_range, output_dir, class_names)

metric['overall'] = {c: overall_AP[idx] for idx, c in enumerate(class_names)}

metric['mAP'] = np.mean(overall_AP)

for th in iou_th_range:

    print("IOU threshold = ", th)

    average_precisions = list(metric[th].values())

    mAP = np.mean(average_precisions)

    print("Average per class mean average precision = ", mAP)

    for class_id in sorted(list(zip(class_names, average_precisions))):

        print(class_id)

    print("_______________________________________________")

print("Overall mean average precision = ", metric['mAP'])
time_start = time.time()

average_precisions = get_average_precisions(gt, pr, class_names, 0.01)

mAP = np.mean(average_precisions)

print("Average per class mean average precision = ", mAP)

for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):

    print(class_id)

print("Time to evaluate = ", time.time() - time_start)  
pr_corr = [b.serialize() for b in pred_box3ds_correct[:10000]]
time_start = time.time()

for iou_threshold in iou_th_range:

    process = Process(target=save_AP, args=(gt, pr_corr, class_names, iou_threshold, output_dir))

    process.start()

    processes.append(process)



for process in processes:

    process.join()

print("Time to evaluate = ", time.time() - time_start)      
# get overall metrics

metric, overall_AP = get_metric_overall_AP(iou_th_range, output_dir, class_names)

metric['overall'] = {c: overall_AP[idx] for idx, c in enumerate(class_names)}

metric['mAP'] = np.mean(overall_AP)

for th in iou_th_range:

    print("IOU threshold = ", th)

    average_precisions = list(metric[th].values())

    mAP = np.mean(average_precisions)

    print("Average per class mean average precision = ", mAP)

    for class_id in sorted(list(zip(class_names, average_precisions))):

        print(class_id)

    print("_______________________________________________")

print("Overall mean average precision = ", metric['mAP'])
time_start = time.time()

average_precisions = get_average_precisions(gt, pr_corr, class_names, 0.01)

mAP = np.mean(average_precisions)

print("Average per class mean average precision = ", mAP)

for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):

    print(class_id)

print("Time to evaluate = ", time.time() - time_start)  