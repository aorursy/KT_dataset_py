# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import glob
import cv2
import numba as nb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from time import time
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from skimage.io import imread
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
np.random.seed(111)
color = sns.color_palette()
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Defining some paths as usual
input_dir = Path('../input/')
data_dir = input_dir / 'data_300x300/data_300x300'
os.listdir(data_dir)
# A function to parse the xmls
def parse_xmls(xml_files):
    data = []
    # Iterate over each file
    for sample in xml_files:
        # Get the xml tree
        tree = ET.parse(sample)

        # Get the root
        root = tree.getroot()

        # Get the members and extract the values
        for member in root.findall('object'):
            # Name of the image file
            filename = root.find('filename').text
            
            # Height and width of the image
            width =  int((root.find('size')).find('width').text)
            height = int((root.find('size')).find('height').text)
            
            # Bounding box coordinates
            bndbox = member.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)
            
            # label to the corresponding bounding box
            label =  member.find('name').text

            data.append((filename, width, height, label, xmin, ymin, xmax, ymax))
    
    # Create a pandas dataframe
    columns_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(data=data, columns=columns_name)

    return df
images = sorted(glob.glob('../input/data_300x300/data_300x300/images/*.jpg'))
xmls = sorted(glob.glob('../input/data_300x300/data_300x300/labels/*.xml'))
print("Total number of images: ", len(images))
print("Total number of xmls: ", len(xmls))
# Parse the xmls and get the data in a dataframe
df = parse_xmls(xmls)
df.head()
# How many classes do we have for object detection?
label_counts = df['class'].value_counts()
print(label_counts)

plt.figure(figsize=(20,8))
sns.barplot(x=label_counts.index, y= label_counts.values, color=color[2])
plt.title('Labels in our dataset', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(label_counts.index)), ['erythrocyte', 'punctate reticulocyte', 'aggregate reticulocyte'])
plt.show()
train, valid = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=111)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
print("Number of training samples: ", len(train))
print("Number of validation samples: ", len(valid))

