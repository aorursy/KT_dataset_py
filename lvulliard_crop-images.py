import os



# There are two ways to load the data from the PANDA dataset:

# Option 1: Load images using openslide

import openslide

# Option 2: Load images using skimage (requires that tifffile is installed)

import skimage.io



# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display

from collections import Counter

import os



import cv2

import skimage.io

from tqdm.notebook import tqdm

import zipfile
# Location of the training images

dataDir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

dataTestDir = '/kaggle/input/prostate-cancer-grade-assessment/test_images'



# Location of training labels

trainLabels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')

testDF = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv').set_index('image_id')



# Output cropped images

cropDir = '/kaggle/working/cropped_train_images/'

cropTestDir = '/kaggle/working/cropped_test_images/'
if not os.path.exists(cropDir):

    os.mkdir(cropDir)

    

if not os.path.exists(cropTestDir):

    os.mkdir(cropTestDir)
# Parameters for cropping images

cropPx= 56

cropN = 16

assert np.sqrt(cropN) == round(np.sqrt(cropN))
def tile(img):

    result = []

    shape = img.shape

    pad0,pad1 = (cropPx - shape[0]%cropPx)%cropPx, (cropPx - shape[1]%cropPx)%cropPx

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=255)

    

    img = img.reshape(img.shape[0]//cropPx,cropPx,img.shape[1]//cropPx,cropPx,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,cropPx,cropPx,3)

    

    if len(img) < cropN:

        img = np.pad(img,[[0,cropN-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:cropN]

    img = img[idxs]

    for i in range(len(img)):

        result.append({'img':img[i], 'idx':i})

    return result
nbCol = int(np.sqrt(cropN))

names = [x.split('.')[0] for x in os.listdir(dataDir)]

for name in tqdm(names):

    img = skimage.io.MultiImage(os.path.join(dataDir+'/',name+'.tiff'))[-1]

    tiles = tile(img)

    stackImg = np.vstack([np.hstack([tiles[nbCol*col + row]['img'] for row in range(nbCol)])

               for col in range(nbCol)])

    cv2.imwrite(cropDir+name+'.png', stackImg)
# If notebook is running on actual test data

if os.path.exists(dataTestDir):

    names = [x.split('.')[0] for x in os.listdir(dataTestDir)]

    for name in tqdm(names):

        img = skimage.io.MultiImage(os.path.join(dataTestDir+'/',name+'.tiff'))[-1]

        tiles = tile(img)

        stackImg = np.vstack([np.hstack([tiles[nbCol*col + row]['img'] for row in range(nbCol)])

                   for col in range(nbCol)])

        cv2.imwrite(cropTestDir+name+'.png', stackImg)