import numpy as np

import os

import shutil

import matplotlib.pyplot as plt

import zipfile

import tensorflow as tf

import xml.etree.ElementTree as ET

from tqdm import tqdm

from keras.models import Model 

from keras.models import Sequential

from keras.layers.core import Dense

from keras.layers.core import Dropout

from keras.layers import Input

from keras.layers import BatchNormalization

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import UpSampling2D

from keras.layers import ReLU

from keras.layers.advanced_activations import LeakyReLU

from keras.initializers import RandomNormal

from keras.optimizers import Adam

from keras import backend as K

from PIL import Image

print(os.listdir("../input"))
ROOT_DIR = '../input/doggos/'

IMAGES_DIR = ROOT_DIR + 'images/Images/'

BREEDS_DIR = ROOT_DIR + 'annotations/Annotation/'

BREEDS = os.listdir(BREEDS_DIR)

IMAGES = []

FOLDERS=[]



'''for r, d, f in os.walk(IMAGES_DIR):

    for folder in d:

        os.listdir(os.path.join(r, folder)))'''

        

# r=root, d=directories, f = files

for r, d, f in os.walk(IMAGES_DIR):

    for file in f:

        if '.jpg' in file:

            IMAGES.append(os.path.join(file))



for r, d, f in os.walk(IMAGES_DIR):

    for folder in d:

        FOLDERS.append(os.path.join(r, folder))

# Summary

print('Total Images: {}'.format(len(IMAGES)))

print('Total Annotations: {}'.format(len(BREEDS)))

print('Total Carpetas de perros: {}'.format(len(FOLDERS)))