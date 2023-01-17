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
import torch

import torchvision

import seaborn as sns

import matplotlib.pyplot as plt



from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET



images_dir = '/kaggle/input/oxford-pets/images/images/'

annotations_dir = '/kaggle/input/oxford-pets/annotations/annotations/xmls/'
sample_image = Image.open('/kaggle/input/oxford-pets/images/images/Abyssinian_109.jpg')



sample_image
with open('/kaggle/input/oxford-pets/annotations/annotations/xmls/Abyssinian_109.xml') as annot_file:

    print(''.join(annot_file.readlines()))
tree = ET.parse('/kaggle/input/oxford-pets/annotations/annotations/xmls/Abyssinian_109.xml')

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