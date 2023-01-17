# Packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import glob

import os, sys

import cv2

from IPython.display import display

plt.rcParams['figure.figsize'] = [7,5]



# Metadata 

data_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

data_df.head()



# Data path

data_path = '/kaggle/input/global-wheat-detection/train/'
# Read a sample image

I = cv2.cvtColor(cv2.imread(data_path + data_df.image_id[0] + '.jpg'), cv2.COLOR_BGR2RGB)

# Show the image

plt.imshow(I);

plt.axis('off')

plt.title('Wheat image')

plt.show()
bboxes = data_df[data_df.image_id==data_df.image_id[0]].bbox.tolist()

J = I.copy()

for i in range(len(bboxes)):

    x = int(str(bboxes[i][1:-1]).split(',')[0][:-2])

    y = int(str(bboxes[i][1:-1]).split(',')[1][1:-2])

    xw = x + int(str(bboxes[i][1:-1]).split(',')[2][1:-2])

    yh = y + int(str(bboxes[i][1:-1]).split(',')[3][1:-2])

    cv2.rectangle(J,(x,y),(xw,yh),(180,190,0),4)

plt.imshow(J);

plt.axis('off')

plt.title('Wheat image bounding boxes')

plt.show()
import torchvision

base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# replace the classifier with a new one, that has num_classes which is user-defined

num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = base_model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)