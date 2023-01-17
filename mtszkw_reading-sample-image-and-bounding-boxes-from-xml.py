import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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
sample_image = Image.open('/kaggle/input/ship-detection/images/boat4.png')



sample_image
with open('/kaggle/input/ship-detection/annotations/boat4.xml') as annot_file:

    print(''.join(annot_file.readlines()))
tree = ET.parse('/kaggle/input/ship-detection/annotations/boat4.xml')

root = tree.getroot()



sample_annotations = []



for neighbor in root.iter('bndbox'):

    xmin = int(neighbor.find('xmin').text)

    ymin = int(neighbor.find('ymin').text)

    xmax = int(neighbor.find('xmax').text)

    ymax = int(neighbor.find('ymax').text)

    

#     print(xmin, ymin, xmax, ymax)

    sample_annotations.append([xmin, ymin, xmax, ymax])

    

print(sample_annotations)
sample_image_annotated = sample_image.copy()



img_bbox = ImageDraw.Draw(sample_image_annotated)



for bbox in sample_annotations:

    print(bbox)

    img_bbox.rectangle(bbox, outline="green") 

    

sample_image_annotated