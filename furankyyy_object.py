# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        kk = os.path.join(dirname, filename)

        kk = cv2.imread(kk)

        print(kk.shape)

        break

    break



# Any results you write to the current directory are saved as output.
import torchvision

from PIL import Image

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms as T

import cv2

import matplotlib.pyplot as plt



# load a model pre-trained pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)



COCO_INSTANCE_CATEGORY_NAMES = [

    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',

    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',

    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',

    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',

    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',

    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',

    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',

    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',

    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',

    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',

    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',

    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'

]



model.eval()
def get_prediction(img_path, threshold):

    img = Image.open(img_path) # Load the image

    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform

    img = transform(img) # Apply the transform to the image

    pred = model([img]) # Pass the image to the model

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score

    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes

    pred_score = list(pred[0]['scores'].detach().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.

    pred_boxes = pred_boxes[:pred_t+1]

    pred_class = pred_class[:pred_t+1]

    return pred_boxes, pred_class
def object_detection_api(img_path, plt_path, threshold=0.8, rect_th=1, text_size=1, text_th=1):



    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions

    img = cv2.imread(plt_path) # Read image with cv2

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    for i in range(len(boxes)):

        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates

        #cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class



    plt.figure(figsize=(20,30)) # display the output image

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.show()
for dirname, _, filenames in os.walk('/kaggle/input/waka-frames'):

    for filename in filenames:

        object_detection_api(os.path.join(dirname, filename),os.path.join('/kaggle/input/outputwaka/',filename))

        img = cv2.imread(os.path.join(dirname, filename))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20,30))

        plt.imshow(img)

        plt.xticks([])

        plt.yticks([])

        plt.show()