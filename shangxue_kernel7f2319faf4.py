# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

from sklearn.utils import shuffle

from tqdm import tqdm_notebook

from glob import glob

from numpy import floor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



def plot_three_samples(letter):

    print("Samples images for letter " + letter)

    base_path = '../input/zhanglabdata/CellData/chest_xray/train/'

    img_path = base_path + letter + '/**'

    path_contents = glob(img_path)

    

    plt.figure(figsize=(16,16))

    imgs = random.sample(path_contents, 3)

    plt.subplot(131)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(imgs[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(imgs[2]))

    return

plot_three_samples('NORMAL')

plot_three_samples('PNEUMONIA')