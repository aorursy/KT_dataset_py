# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
import torch

import torchvision

import seaborn as sns

import matplotlib.pyplot as plt



from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET



images_dir = '/kaggle/input/oxford-pets/images/images/'

annotations_dir = '/kaggle/input/oxford-pets/annotations/annotations/xmls/'
sample_id = 146



sample_image_path = f'/kaggle/input/oxford-pets/images/images/Abyssinian_{sample_id}.jpg'

sample_annot_path = f'/kaggle/input/oxford-pets/annotations/annotations/xmls/Abyssinian_{sample_id}.xml'
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
cat_id = 17

cat_boxes = [x.detach().numpy().tolist() for i, x in enumerate(result[0]['boxes']) if result[0]['labels'][i] == cat_id]

cat_boxes
sample_image_annotated = sample_image.copy()



img_bbox = ImageDraw.Draw(sample_image_annotated)



for bbox in sample_annotations:

    img_bbox.rectangle(bbox, outline="white") 



for bbox in cat_boxes:

    x1, x2, x3, x4 = map(int, bbox)

    print(x1, x2, x3, x4)

    img_bbox.rectangle([x1, x2, x3, x4], outline="red") 



sample_image_annotated