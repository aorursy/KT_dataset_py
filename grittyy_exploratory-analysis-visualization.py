import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import dicom

import cv2

from skimage import data, io, filters

import os

import matplotlib.pyplot as plt

import pylab



%config InlineBackend.figure_format = 'retina'
def show(slice):

    plt.imshow(slice, cmap=plt.cm.bone)
%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (10.0, 25.0)



for i in range(96):

    plt.subplot(16,6,i+1)

    show(full_img[int(full_img.shape[0]/96*i),:,:])    

    plt.xticks([])

    plt.yticks([])