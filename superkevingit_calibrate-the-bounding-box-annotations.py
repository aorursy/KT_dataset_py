import pandas as pd

import re

import numpy as np

from collections import namedtuple

from typing import List, Union

from matplotlib import pyplot as plt

import cv2

import json





DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
# define some functions, some of them were modified from other kernels

def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r





def transform_data_struct(train_df):

    # transform data structure

    train_df['x'] = -1

    train_df['y'] = -1

    train_df['w'] = -1

    train_df['h'] = -1



    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

    #train_df.drop(columns=['bbox'], inplace=True)

    train_df['x'] = train_df['x'].astype(np.float)

    train_df['y'] = train_df['y'].astype(np.float)

    train_df['w'] = train_df['w'].astype(np.float)

    train_df['h'] = train_df['h'].astype(np.float)

    return train_df
def overlap(gt: List[Union[int, float]],

                  pred: List[Union[int, float]],

                  form: str = 'pascal_voc') -> float:

    """Calculates the overlap percentage.



    Args:

        gt: List[Union[int, float]] coordinates of the ground-truth box

        pred: List[Union[int, float]] coordinates of the prdected box

        form: str gt/pred coordinates format

            - pascal_voc: [xmin, ymin, xmax, ymax]

            - coco: [xmin, ymin, w, h]

    Returns:

        IoU: float Intersection over union (0.0 <= iou <= 1.0)

    """

    Box = namedtuple('Box', 'xmin ymin xmax ymax')



    if form == 'coco':

        bgt = Box(gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])

        bpr = Box(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])

    else:

        bgt = Box(gt[0], gt[1], gt[2], gt[3])

        bpr = Box(pred[0], pred[1], pred[2], pred[3])



    overlap_area = 0.0



    # Calculate overlap area

    dx = min(bgt.xmax, bpr.xmax) - max(bgt.xmin, bpr.xmin)

    dy = min(bgt.ymax, bpr.ymax) - max(bgt.ymin, bpr.ymin)



    if (dx > 0) and (dy > 0):

        overlap_area = dx * dy



    return overlap_area
def plot_image_bboxes(img_id: str = '', box: list = None):

    box = [int(x) for x in box]

    image = cv2.imread(f'{DIR_TRAIN}/{img_id}.jpg', cv2.IMREAD_COLOR)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.set_axis_off()

    ax.imshow(image / 255)
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df = transform_data_struct(train_df)
seg_list = []

cnt_max_num = 2

cnt_ratio = 0.8

total_img_num = len(train_df['image_id'].unique())

train_cnt = 0

error_img_id = []



for img_id in train_df['image_id'].unique():

    train_cnt += 1

    print('processing on '+img_id+'('+str(train_cnt)+'/'+str(total_img_num)+')')

    img_df = train_df[train_df['image_id']==img_id]

    for index_1, detect_df in img_df.iterrows():

        cnt = 0

        detect_bb = [detect_df['x'], detect_df['y'], detect_df['w'], detect_df['h']]

        for _, test_df in img_df.iterrows():

            test_bb = [test_df['x'], test_df['y'], test_df['w'], test_df['h']]

            overlap_area = overlap(detect_bb, test_bb, 'coco')

            if overlap_area > test_bb[2] * test_bb[3] * cnt_ratio:

                cnt += 1

            if cnt > cnt_max_num:

                print(img_id)

                print(detect_bb)

                error_img_id.append({'img_id':img_id, 'index': index_1, 'detect_bb': detect_bb})

                img_df = img_df.drop(labels=index_1, axis=0)

                break

    seg_list.append(img_df)
# save

pd.concat(seg_list).to_csv('calibrate_train.csv')

with open('error_img_id.json', 'w') as f:

    json.dump(error_img_id, f)
# vis

for im in error_img_id:

    img_id = im['img_id']

    box = im['detect_bb']

    plot_image_bboxes(img_id, box)
error_img_id

len(error_img_id)