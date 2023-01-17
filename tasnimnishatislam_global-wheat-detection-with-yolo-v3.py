import pandas as pd

import numpy as np

import cv2

import os

import re

import shutil

from tqdm import tqdm

from zipfile import ZipFile



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
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape
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
train_df.head()
train_df['x_center'] = train_df['x'] + train_df['w']/2.0
train_df.head()
train_df['y_center'] = train_df['y'] + train_df['h']/2.0
train_df.head()
image_ids = train_df['image_id'].unique()

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
def generate_txt_file(dataframe):

    folder = "Hello"

    DIR_PATH = folder + ".zip"

    zipObj = ZipFile(DIR_PATH, 'w')

    for i in tqdm(range(len(dataframe))):

        label = DIR_PATH + dataframe.iloc[i]['image_id'] + ".txt"

        f = open(label, "w")

        line = "{} {} {} {} {}\n".format(0, dataframe.iloc[i]['x_center'], dataframe.iloc[i]['y_center'], dataframe.iloc[i]['w'], dataframe.iloc[i]['h'])

        f.write(line)

        f.close()

        zipObj.write(label)  

    

    zipObj.close()
generate_txt_file(train_df)