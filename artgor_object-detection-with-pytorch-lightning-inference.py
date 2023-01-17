from __future__ import print_function

from PIL import Image

from albumentations.core.composition import Compose

from collections import defaultdict, deque

from itertools import product

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from torchvision.transforms import functional as F

from typing import Any, Dict, List

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import ast

import copy

import cv2

import datetime

import errno

import importlib

import json

import numpy as np

import os

import pandas as pd

import pickle

import random

import shutil

import tempfile

import time

import torch

import torch._six

import torch.distributed as dist

import torch.utils.data

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt

%matplotlib inline
def set_seed(seed: int = 666):

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    

def get_test_dataset():

    """

    Get test dataset



    Args:

        cfg:



    Returns:



    """



    test_img_dir = f'/kaggle/input/global-wheat-detection/test'



    valid_augs = A.Compose([ToTensorV2(always_apply=True, p=1.0)],

                           p=1.0,

                           bbox_params={'format': 'pascal_voc', 'label_fields': ['labels'], 'min_area': 0.0, 'min_visibility': 0.0},

                           keypoint_params=None,

                           additional_targets={})



    test_dataset = WheatDataset(None,

                                 'test',

                                 test_img_dir,

                                 valid_augs)



    return test_dataset



class WheatDataset(Dataset):



    def __init__(self,

                 dataframe: pd.DataFrame = None,

                 mode: str = 'train',

                 image_dir: str = '',

                 transforms: Compose = None):

        """

        Prepare data for wheat competition.



        Args:

            dataframe: dataframe with image id and bboxes

            mode: train/val/test

            image_dir: path to images

            transforms: albumentations

        """

        self.image_dir = image_dir

        self.df = dataframe

        self.mode = mode

        self.image_ids = os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()

        self.transforms = transforms



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx].split('.')[0]

        # print(image_id)

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)



        # normalization.

        # TO DO: refactor preprocessing

        image /= 255.0



        # test dataset must have some values so that transforms work.

        target = {'labels': torch.as_tensor([[0]], dtype=torch.float32),

                  'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)}



        # for train and valid test create target dict.

        if self.mode != 'test':

            image_data = self.df.loc[self.df['image_id'] == image_id]

            boxes = image_data[['x', 'y', 'x1', 'y1']].values



            areas = image_data['area'].values

            areas = torch.as_tensor(areas, dtype=torch.float32)



            # there is only one class

            labels = torch.ones((image_data.shape[0],), dtype=torch.int64)

            iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)



            target['boxes'] = boxes

            target['labels'] = labels

            target['image_id'] = torch.tensor([idx])

            target['area'] = areas

            target['iscrowd'] = iscrowd



            if self.transforms:

                image_dict = {

                    'image': image,

                    'bboxes': target['boxes'],

                    'labels': labels

                }

                image_dict = self.transforms(**image_dict)

                image = image_dict['image']

                target['boxes'] = torch.as_tensor(image_dict['bboxes'], dtype=torch.float32)



        else:

            image_dict = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': target['labels']

            }

            image = self.transforms(**image_dict)['image']



        return image, target, image_id



    def __len__(self) -> int:

        return len(self.image_ids)

    

    

def collate_fn(batch):

    return tuple(zip(*batch))





def format_prediction_string(boxes, scores):

    pred_strings = []

    for s, b in zip(scores, boxes.astype(int)):

        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')



    return " ".join(pred_strings)
set_seed(42)



test_dataset = get_test_dataset()



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

device = 'cuda'

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# Load the trained weights

model.load_state_dict(torch.load('/kaggle/input/object-detection-with-pytorch-lightning/fasterrcnn_resnet50_fpn.pth'))

model.cuda()

model.eval()



test_loader = torch.utils.data.DataLoader(test_dataset,

                                          batch_size=4,

                                          num_workers=2,

                                          shuffle=False,

                                          collate_fn=collate_fn)

detection_threshold = 0.5

results = []



for images, _, image_ids in test_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()



        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]



        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)