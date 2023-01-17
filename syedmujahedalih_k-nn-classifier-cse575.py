# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import matplotlib.image as mpimg

import os

import pandas as pd
DATASET_PATH = "/kaggle/input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/"

print(os.listdir(DATASET_PATH))
df_images = pd.read_csv('../input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/images.csv')
df_images.head()
import torch 

import torchvision

import PIL

import os

import sys

from PIL import Image

from torchvision import models

from torchvision import transforms
script_dir = os.path.dirname('/kaggle/output/')
print(script_dir)
imageTransformation = transforms.Compose([            

transforms.Resize(256),      # obtaining a resize of the image as a 256X256 Image              

transforms.CenterCrop(224),  # center crop applied to the image to botain a 224X224 Image from the center             

transforms.ToTensor(),       # obtaining the image tensor              

transforms.Normalize(        # normalising is done on the tensor of the image. # values represent the three colros RGB              

mean=[0.485, 0.456, 0.406],                

std=[0.229, 0.224, 0.225]                 

)])
img = Image.open("/kaggle/input/fashion-product-images-dataset/fashion-dataset/images/38440.jpg")
plt.imshow(img)
imgTensor = imageTransformation(img)
imgTensor