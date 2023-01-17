# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()
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
from PIL import Image

import torchvision.transforms as T

import cv2

import matplotlib.pyplot as plt
def get_prediction(img_path,threshold):

    # Load the image

    img = Image.open(img_path)

    # Define PyTorch Transform

    transform = T.Compose([T.ToTensor()])

    # Apply the transform to the image

    img = transform(img)

    # Pass the image to the model

    pred = model([img])

    #Get the Prediction Score

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]

    #Bounding boxes

    pred_boxes = [[(i[0],i[1]),(i[2],i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

    pred_score = list(pred[0]['scores'].detach().numpy())

    # Get list of index with score greater than threshold.

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    pred_boxes = pred_boxes[:pred_t+1]

    pred_class = pred_class[:pred_t+1]

    return pred_boxes, pred_class
def object_detection_api(img_path,threshold=0.5, rect_th=3, text_size = 3, text_th = 3):

    # Get predictions

    boxes,pred_cls = get_prediction(img_path,threshold)

    # Read image with cv2

    img = cv2.imread(img_path)

    #Convert to RGB

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):

        # Draw Rectangle with the coordinates

        cv2.rectangle(img, boxes[i][0], boxes[i][1], color = (0, 255, 0),thickness = rect_th)

        # Write the prediction class

        cv2.putText(img,pred_cls[i],boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,text_size, (0,255,0),thickness = text_th)

    # display the output image

    plt.figure(figsize = (20,30))

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.show()
!wget https://www.wsha.org/wp-content/uploads/banner-diverse-group-of-people-2.jpg -O people.jpg
object_detection_api('./people.jpg', threshold = 0.8)
!wget https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg -O car.jpg

 

object_detection_api('./car.jpg', rect_th=6, text_th=20, text_size=5)
!wget https://cdn.pixabay.com/photo/2013/07/05/01/08/traffic-143391_960_720.jpg -O traffic.jpg

 

object_detection_api('./traffic.jpg', rect_th=2, text_th=1, text_size=1)
!wget https://images.unsplash.com/photo-1458169495136-854e4c39548a -O girl_cars.jpg

 

object_detection_api('./girl_cars.jpg', rect_th=15, text_th=7, text_size=5, threshold=0.8)
!wget https://live.staticflickr.com/4238/34598338584_f40017c704_b.jpg -O 34598338584_f40017c704_b.jpg



object_detection_api('./34598338584_f40017c704_b.jpg',rect_th = 2, text_th = 2,text_size = 0.6)