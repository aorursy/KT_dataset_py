import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch

import os

import time

import shutil

import torch.nn as nn

from skimage import io

import torchvision

import cv2

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler

from albumentations.pytorch import ToTensor

from torchvision import utils

from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,

                            Compose, GaussNoise)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

csv_path = '/kaggle/input/global-wheat-detection/train.csv'

test_dir = '/kaggle/input/global-wheat-detection/test'
def get_transforms():

            list_transforms = []

            list_transforms.extend(

                    [

        

            ToTensor(),

                    ])

            list_trfms = Compose(list_transforms)

            return list_trfms
class Wheatset(Dataset):

    def __init__(self,image_dir):

        super().__init__()

   

        self.image_dir = image_dir

        self.images = os.listdir(image_dir)

        self.transforms = get_transforms()

        

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self,idx):

        image = self.images[idx]

        image_arr = io.imread(os.path.join(self.image_dir,image))

        image_id = str(image.split('.')[0])

        

        if self.transforms:

            sample = {

                'image': image_arr,

            }

            sample = self.transforms(**sample)

            image = sample['image']

               

        return image, image_id
def collate_fn(batch):

    return tuple(zip(*batch))

test_dir = '/kaggle/input/global-wheat-detection/test'



test_dataset = Wheatset(test_dir)



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
device = torch.device('cpu')





# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    

num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# Load the trained weights

weights = '/kaggle/input/faster-rcnn-with-psudo-labeling-pytorch/psudomodel.pt'



checkpoint = torch.load(weights,map_location=device)

model.load_state_dict(checkpoint['state_dict'])

model.eval()



x = model.to(device)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)









detection_threshold = 0.5

results = []

for images, image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)

    

    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }

     

        results.append(result)

final_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

final_df.head()
final_df.to_csv('submission.csv', index=False)