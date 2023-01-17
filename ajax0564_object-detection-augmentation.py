# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
import cv2 

import os

BASE_DIR = '/kaggle/input/global-wheat-detection'

image_id = '7264c58b9'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = image/255

plt.figure(figsize = (10, 10))

plt.imshow(image)

plt.show()
boxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.int32).values
fig, ax = plt.subplots(nrows=1, figsize=(8, 8))



get_bbox(boxes, ax, color='red', bbox_format='coco')

ax.title.set_text('Image')

ax.imshow(image)

plt.show()
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.VerticalFlip(1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('verticle flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.HorizontalFlip(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.ShiftScaleRotate(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('shift scale rotate')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomBrightness(p=1,limit = 0.9),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomContrast(p=1,limit = 0.9),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.GaussianBlur(p=1,blur_limit = 14),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.ChannelShuffle(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomSnow(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomRain(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomFog(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([

        A.Resize(1024, 1024),   

        A.RandomShadow(p=1),

       ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([A.RandomSizedCrop(min_max_height=(600, 600), height=1024, width=1024, p=0.5), ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()
aug = A.Compose([A.Cutout(num_holes=10, max_h_size=64, max_w_size=64, fill_value=0, p=0.5), ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



aug_result = aug(image=image, bboxes=boxes, labels=np.ones(len(boxes)))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('horizontal flip')

ax[1].imshow(aug_result['image'])

plt.show()