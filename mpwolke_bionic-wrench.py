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
# importing all the required libraries

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import skimage.io as io

from skimage.transform import rotate, AffineTransform, warp

from skimage.util import random_noise

from skimage.filters import gaussian

import matplotlib.pyplot as plt

import PIL.Image

import matplotlib.pyplot as plt

import torch

from torchvision import transforms
def imshow(img, transform):

    """helper function to show data augmentation

    :param img: path of the image

    :param transform: data augmentation technique to apply"""

    

    img = PIL.Image.open(img)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].set_title(f'original image {img.size}')

    ax[0].imshow(img)

    img = transform(img)

    ax[1].set_title(f'transformed image {img.size}')

    ax[1].imshow(img)
loader_transform = transforms.Resize((140, 140))



imshow('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Wrench/000024.jpg', loader_transform)
loader_transform = transforms.CenterCrop(140)

imshow('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Wrench/000024.jpg', loader_transform)
from fastai.vision import *
tfms = get_transforms(max_rotate=25)