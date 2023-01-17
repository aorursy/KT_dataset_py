import os

import numpy as np

import pandas as pd



import torch

import torchvision



import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET



images_dir = '/kaggle/input/ship-detection/images/'

annotations_dir = '/kaggle/input/ship-detection/annotations/'



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
sample_id = 44



sample_image_path = f'/kaggle/input/ship-detection/images/boat{sample_id}.png'

sample_annot_path = f'/kaggle/input/ship-detection/annotations/boat{sample_id}.xml'
sample_image = Image.open(sample_image_path)

sample_image
with open(sample_annot_path) as annot_file:

    print(''.join(annot_file.readlines()))
tree = ET.parse(sample_annot_path)

root = tree.getroot()



sample_annotations = []



for neighbor in root.iter('bndbox'):

    xmin = int(neighbor.find('xmin').text)

    ymin = int(neighbor.find('ymin').text)

    xmax = int(neighbor.find('xmax').text)

    ymax = int(neighbor.find('ymax').text)

    

    sample_annotations.append([xmin, ymin, xmax, ymax])

    

print('Ground-truth annotations:', sample_annotations)
sample_image_annotated = sample_image.copy()



img_bbox = ImageDraw.Draw(sample_image_annotated)



for bbox in sample_annotations:

    img_bbox.rectangle(bbox, outline="white") 

    

sample_image_annotated
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(

    pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)



model
model.eval()



np_sample_image = np.array(sample_image.convert("RGB"))



transformed_img = torchvision.transforms.transforms.ToTensor()(

        torchvision.transforms.ToPILImage()(np_sample_image))



result = model([transformed_img])



result
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

    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
boat_id = 9

boat_boxes = [x.detach().numpy().tolist() for i, x in enumerate(result[0]['boxes']) if result[0]['labels'][i] == boat_id]

boat_boxes
sample_image_annotated = sample_image.copy()



img_bbox = ImageDraw.Draw(sample_image_annotated)



for bbox in sample_annotations:

    img_bbox.rectangle(bbox, outline="white") 



for bbox in boat_boxes:

    x1, x2, x3, x4 = map(int, bbox)

    print(x1, x2, x3, x4)

    img_bbox.rectangle([x1, x2, x3, x4], outline="red") 



sample_image_annotated