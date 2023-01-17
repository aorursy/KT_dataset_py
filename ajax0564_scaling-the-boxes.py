import numpy as np # linear algebra

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



import matplotlib.patches as patches

def get_bbox(bboxes, col, color='red', bbox_format='coco'):

    

    for i in range(len(bboxes)):

            rect = patches.Rectangle(

                (bboxes[i][0], bboxes[i][1]),

                bboxes[i][2], 

                bboxes[i][3], 

                linewidth=2, 

                edgecolor=color, 

                facecolor='none')



            col.add_patch(rect)
import re 

train_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns=['bbox'], inplace=True)

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)



train_df.head(5)
'''3687

117344

173

113947

52868

2159

2169

121633

121634

147504

118211

52727

147552'''
for k in [3687,117344,173,113947,52868,2159,2169,121633,121634,147504,118211,52727,147552]:

    print(train_df.iloc[k][0])
import cv2 

import os

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '893938464'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df.loc[147552,'x']
train_df.loc[147552,'x'] = 166+296

train_df.loc[147552,'w'] = 148

train_df.iloc[147552]
import cv2 

import os

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '893938464'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df.loc[147552,'x'] = 462

train_df.iloc[147552]
import cv2 

import os

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '893938464'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
# import cv2# import os

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '9a30dd802'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df.iloc[52868]
train_df.loc[52868,'w'] = 417-212

train_df.loc[52868,'y'] = 301+185

train_df.loc[52868,'h'] = 370-185





BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '9a30dd802'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = 'd7a02151d'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df = train_df.drop([118211],axis = 0)

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = 'd7a02151d'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()


BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '42e6efaaa'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df = train_df.drop([113947],axis = 0)

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '42e6efaaa'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()


BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '41c0123cc'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
train_df.iloc[173]
train_df.loc[173,'x'] = 300

train_df.loc[173,'y'] = 15+246

train_df.loc[173,'w'] = 68

train_df.loc[173,'h'] = 130



BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '41c0123cc'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()


BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = 'dcafcae79'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()