import os

import pydicom as dcm

import glob

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

import torch

from sklearn.linear_model import Ridge

import random

from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import mean_squared_error

import category_encoders as ce

from PIL import Image

import cv2

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")



import seaborn as sns

p = sns.color_palette()

import plotly.express as px
def dicom_to_image(filename):

    im = dcm.dcmread(filename)

    img = im.pixel_array

    img[img == -2000] = 0

    return img
files_darkimages = glob.glob('../input/osic-pulmonary-fibrosis-progression/train/ID00105637202208831864134/*.dcm')

print("Patiend files :",len(files_darkimages))

f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image(files_darkimages[i]), cmap=plt.cm.bone)
from skimage import exposure

def dicom_to_image2(filename):

    im = dcm.dcmread(filename)

    img = im.pixel_array

    img = exposure.equalize_hist(img)

    return img



def dicom_to_image3(filename):

    im = dcm.dcmread(filename)

    img = im.pixel_array

    img = exposure.equalize_adapthist(img)

    return img
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image2(files_darkimages[i]), cmap=plt.cm.bone)

plt.title("Dark images fixed using equalize_hist")
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image3(files_darkimages[i]), cmap=plt.cm.bone)

plt.title("Dark images fixed using equalize_adapthist")