import pandas as pd

import numpy as np

import cv2

import os

import re



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'



#a way to join names of the file given



train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape



#read the train.csv

#it has 5 columns and ogonito items
train_df.head()
train_df['x'] = -1

train_df.head()

#it actually assigns a new col and also a single value to the entire col
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



#they are defining a bbox by this
r = np.array(re.findall("([0-9]+[.]?[0-9]*)", "[834.0, 222.0, 56.0, 36.0]"))

print(r)



#get all the individual values
def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



#if there is no value then [-1, -1, -1, -1 is assigned]
#seperate all the values from bbox

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

#remove the "bbox" column from the dataframe permanently

train_df.drop(columns=['bbox'], inplace=True)

 

#assign the value as float type, that was extracted from bbox as string type

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
train_df.head()
#got all the unique image ids

image_ids = train_df['image_id'].unique()

#randomly divided this as training and validation sets

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape, train_df.shape