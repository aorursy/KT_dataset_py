import numpy as np 

import pandas as pd
import imageio
import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps

import scipy.ndimage as ndi
import os

print(os.listdir("../input"))
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Activation, ZeroPadding2D, BatchNormalization

from keras.optimizers import Adam

from sklearn.metrics import accuracy_score

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing import image

from keras.utils import plot_model
dirname = '/kaggle/input'

train_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/train')

train_nrml_pth = os.path.join(train_path, 'NORMAL')

train_pnm_pth = os.path.join(train_path, 'PNEUMONIA')

test_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

test_nrml_pth = os.path.join(test_path, 'NORMAL')

test_pnm_pth = os.path.join(test_path, 'PNEUMONIA')

val_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

val_nrml_pth = os.path.join(val_path, 'NORMAL')

val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')
dirname_work = '/kaggle'

dir_chest_xray = os.path.join('/kaggle', 'chest_xray')

os.mkdir('/kaggle/chest_xray/')

os.mkdir('/kaggle/chest_xray/train')

os.mkdir('/kaggle/chest_xray/train/NORMAL')

os.mkdir('/kaggle/chest_xray/train/PNEUMONIA')

train_path_work = os.path.join(dir_chest_xray, 'train')

train_nrml_pth_work = os.path.join(train_path_work, 'NORMAL')

train_pnm_pth_work = os.path.join(train_path_work, 'PNEUMONIA')





os.mkdir('/kaggle/chest_xray/test')

os.mkdir('/kaggle/chest_xray/test/NORMAL')

os.mkdir('/kaggle/chest_xray/test/PNEUMONIA')

test_path_work = os.path.join(dir_chest_xray, 'test')

test_nrml_pth_work = os.path.join(test_path_work, 'NORMAL')

test_pnm_pth_work = os.path.join(test_path_work, 'PNEUMONIA')
def image_resizing(path_from, path_to, height=500, width=500):

    size = height, width

    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = img.resize(size, Image.ANTIALIAS)

            img = img.convert("RGB")

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
image_resizing(train_nrml_pth, train_nrml_pth_work, 300, 300)
image_resizing(train_pnm_pth, train_pnm_pth_work, 300, 300)
image_resizing(test_nrml_pth, test_nrml_pth_work, 300, 300)

image_resizing(test_pnm_pth, test_pnm_pth_work, 300, 300)