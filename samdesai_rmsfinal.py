# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import tensorflow 



import keras

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.optimizers import Adam

from keras.optimizers import RMSprop

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.preprocessing import image

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

from keras.models import load_model



import matplotlib.pyplot as plt

import itertools

from matplotlib import pyplot as plt

import glob

import matplotlib.image as mpimg

from keras.preprocessing import image

import imageio as im

import glob

import matplotlib
test_path = '/kaggle/input/plantvillagepredictions/val/val'
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',

                                                                                                    'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',

                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',

                                                                                                    'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10,shuffle=False)
from keras.models import load_model
model = Sequential()
model = load_model('/kaggle/input/rmsmodels/Final_RMS.h5')
predicted_label = model.predict_generator(test_batches,steps=440,verbose=0)
true_labels = test_batches.classes
labels = np.zeros((4396,1))

count = 0

for i in range(0,4396):

    for j in range(0,38):

        if predicted_label[i][j] == max(predicted_label[i]):

            labels[i] = j

            if j == true_labels[i]:

                count+=1

print(count)

print(count/4396)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# No Batch-Normalization and No Dropout + Augmentation "CustomModel.h5"

print(f1_score(labels,true_labels, average="macro"))

print(precision_score(labels,true_labels, average="macro"))

print(recall_score(labels,true_labels, average="macro"))  