import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import PIL
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
import random
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import save
from numpy import load

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten

X = load('G:/Desktop/intel image classification/data224(3).npy')
y = load('G:/Desktop/intel image classification/y224(3)(all).npy')
x_test = load('G:/Desktop/intel image classification/data224(3)_test.npy')
y_test = load('G:/Desktop/intel image classification/y224(3)(all)_test.npy')
IMG_SHAPE = (244, 244, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
x = base_model.get_layer('block4_pool').output
prediction = tf.keras.layers.Flatten()(x)
model = Model(inputs=base_model.input, outputs=prediction)
%%time
block4_pool_features = model.predict(X)
block4_pool_features_test = model.predict(x_test)
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
block4_pool_features.shape
block4_pool_features_test.shape
%%time
svm = LinearSVC()
svm.fit(block4_pool_features, y)
%%time
preds_train = svm.predict(block4_pool_features)
preds_test = svm.predict(block4_pool_features_test)
accuracy_score(y, preds_train), accuracy_score(y_test, preds_test)











