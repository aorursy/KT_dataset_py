import os

import shutil



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



import cv2

import skimage.io

from tqdm.notebook import tqdm



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.test.is_gpu_available()
# Location of the training images

dataDir = '/kaggle/input/crop-images/cropped_train_images/'



# Location of training labels

trainLabels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')

testDF = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv').set_index('image_id')



# Output cropped images

cropDir = '/kaggle/working/cropped_train_images/'



inputShape = (224, 224, 3)

epochs = 90
# How many train objects should be included in one batch (higher = faster but less accurate)

# Take care that the batch size is smaller than the amount of total images analyzed

batchSize = 256

INIT_LR = 1e-3
trainDatagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, validation_split = 0.1,

                                  zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
trainDF = pd.DataFrame(list(zip(trainLabels.index + ".png", trainLabels.isup_grade.astype(str))), 

               columns =['x_col', 'y_col']) 



# Uncomment the following if the assumption needs to be re-checked

# for x in trainDF.x_col:

#     assert x in os.listdir(dataDir)
trainGenerator = trainDatagen.flow_from_dataframe(

    trainDF, x_col="x_col", y_col="y_col",

    directory=dataDir,  # this is the target directory

    batch_size=batchSize,

    class_mode = "sparse",

    subset="training",

    target_size=(inputShape[0], inputShape[1]),

    color_mode='rgb')
valGenerator = trainDatagen.flow_from_dataframe(

    trainDF, x_col="x_col", y_col="y_col",

    directory=dataDir,  # this is the target directory

    batch_size=batchSize,

    class_mode = "sparse",

    subset="validation",

    target_size=(inputShape[0], inputShape[1]),

    color_mode='rgb')
from keras.applications import Xception

from keras.models import Sequential

from keras.layers import Dense, Flatten, GlobalAveragePooling2D
numClasses = len(set(trainLabels.isup_grade))

weightFile = "/kaggle/input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"



myModel = Sequential()

myModel.add(Xception(include_top=False, pooling='avg', weights=weightFile))

myModel.add(Dense(numClasses, activation='softmax'))



# Say not to train first layer (Xception) model. It is already trained

myModel.layers[0].trainable = False
myModel.summary()
# Optimaztion function

opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)



myModel.compile(loss="sparse_categorical_crossentropy",

              optimizer=opt,

              metrics=["binary_accuracy"])
H = myModel.fit_generator(trainGenerator,

                        steps_per_epoch=len(trainLabels.index) // batchSize,

                        epochs=epochs, 

                        validation_data=valGenerator,

                        validation_steps=1,

                        verbose=1)
# Save model

myModel.save('/kaggle/working/Xception_'+str(epochs)+'.model')