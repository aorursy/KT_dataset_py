import numpy as np

import pandas as pd

from tensorflow.keras.models import load_model

import cv2

import skimage.io

import os

import sys

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.metrics import Metric

import tensorflow.keras.backend as K

sys.path.insert(0, '/kaggle/input/efficientnet-keras-source-code/')

import efficientnet.tfkeras as efn
img_size = 512

test_dir = '/kaggle/input/prostate-cancer-grade-assessment/test_images'

model = load_model('../input/pandas-training-file/model.h5', compile=False)
def get_image(img_name):

    img_path = os.path.join(test_dir, f'{img_name}.tiff')    

    img = skimage.io.MultiImage(img_path)        

    img = cv2.resize(img[-1], (img_size, img_size))    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255

    return img
sub = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv')

test_images = sub['image_id'].values
labels = []

try:

    for image in test_images:                

        img = get_image(image)    

        im1 = img.reshape((1, img_size, img_size, 3))    

        preds = model.predict(im1, batch_size=1)    

        result = np.argmax(preds ,axis = 1)               

        labels.append(result)

    sub['isup_grade'] = labels

except:

    print('exception')
sub['isup_grade'] = sub['isup_grade'].astype(int)

sub.to_csv('submission.csv', index=False)

sub.head()