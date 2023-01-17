import tensorflow as tf

import os

import numpy as np

import random

from tqdm import tqdm

from skimage.io import imread, imshow

from skimage.transform import resize

import matplotlib.pyplot as plt

from zipfile import ZipFile
TRAIN_PATH = "../input/data-science-bowl-2018/stage1_train.zip"

TEST_PATH = "../input/data-science-bowl-2018/stage1_test.zip"
!mkdir /kaggle/working/train
!mkdir /kaggle/working/test
with ZipFile("../input/data-science-bowl-2018/stage1_train.zip") as z:

    z.extractall("/kaggle/working/train/")

with ZipFile("../input/data-science-bowl-2018/stage1_test.zip") as z:

    z.extractall("/kaggle/working/test/")
