import pandas as pd

import numpy as np

import cv2

import re

import os

import matplotlib.pyplot as plt

import matplotlib.patches as patches

%matplotlib inline
def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")



df['x'] = -1

df['y'] = -1

df['w'] = -1

df['h'] = -1



df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))



df['x'] = df['x'].astype(float)

df['y'] = df['y'].astype(float)

df['w'] = df['w'].astype(float)

df['h'] = df['h'].astype(float)



df.drop(columns=['bbox'], inplace=True)

df['x1'] = df['x'] + df['w']

df['y1'] = df['y'] + df['h']

df['area'] = df['w'] * df['h']



df["image_id"] = df["image_id"] + ".jpg"



df.head()
for cols in df[['x','y','w','h','x1','y1', 'area']].columns:

    print(f"min of {cols} column = {np.min(df[cols])} and max of {cols} column = {np.max(df[cols])}")
small_boxes = df[df['area'] < 50]

small_boxes = small_boxes[['image_id','x','y','w','h','x1','y1','area']]

small_boxes
size = len(small_boxes['image_id'].unique())

fig, ax = plt.subplots(nrows = size, ncols = 1, figsize=(50, 50))

for i, img_idx in enumerate(small_boxes['image_id'].unique()):

    images = cv2.imread(os.path.join("/kaggle/input/global-wheat-detection/train", img_idx), cv2.IMREAD_COLOR)

    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)

    images /= 255.0

    data = small_boxes[small_boxes['image_id'] == img_idx]

    bbox = data[['x', 'y', 'x1', 'y1']].values

    area = data['area'].values.item()

    for box in bbox:

        ax[i,].add_patch(

            patches.Rectangle(

            (box[0], box[1]),

            box[2]-box[0],

            box[3]-box[1],

            linewidth=2,

            fill=False,

            color='red'))

        ax[i,].set_axis_off()

        ax[i,].set_title(f"Image with very small bounding box id = {img_idx}, area = {area}")

        ax[i,].imshow(images)

plt.show()

plt.close()
large_boxes = df[df['area'] > 160000] 

large_boxes = large_boxes[['image_id','x','y','w','h','x1','y1','area']]

large_boxes
size = len(large_boxes['image_id'].unique())

fig, ax = plt.subplots(nrows = size, ncols = 1, figsize=(50, 50))

for i, img_idx in enumerate(large_boxes['image_id'].unique()):

    images = cv2.imread(os.path.join("/kaggle/input/global-wheat-detection/train", img_idx), cv2.IMREAD_COLOR)

    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)

    images /= 255.0

    data = large_boxes[large_boxes['image_id'] == img_idx]

    bbox = data[['x', 'y', 'x1', 'y1']].values

    area = data['area'].values.item()

    for box in bbox:

        ax[i,].add_patch(

            patches.Rectangle(

            (box[0], box[1]),

            box[2]-box[0],

            box[3]-box[1],

            linewidth=2,

            fill=False,

            color='red'))

        ax[i,].set_axis_off()

        ax[i,].set_title(f"Image with very large bounding box id = {img_idx}, area = {area}")

        ax[i,].imshow(images)

plt.show()

plt.close()