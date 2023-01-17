import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import utils

import os







# Networks

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16



from keras.preprocessing.image import ImageDataGenerator



# Layers

from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import load_model





# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime



print(os.listdir("../input/age-dataset/data2/data2/"))



# Any results you write to the current directory are saved as output.
BATCH_SIZE = 28

WIDTH = 299

HEIGHT = 299

FC_LAYERS = [1024, 1024]

TRAIN_DIR = "../input/age-dataset/data2/data2/train"

VAL_DIR = "../input/age-dataset/data2/data2/val"
preprocessing_function = None

base_model = None

from keras.applications.vgg16 import preprocess_input

preprocessing_function = preprocess_input

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
train_datagen =  ImageDataGenerator(

      preprocessing_function=preprocessing_function

)

val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)



train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)



validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

class_list = utils.get_subfolders(TRAIN_DIR)

utils.save_class_list(class_list, model_name="VGG16", dataset_name="dataset")



finetune_model = utils.build_finetune_model(base_model, dropout=0.0001, fc_layers=FC_LAYERS, num_classes=len(class_list))



adam = Adam(lr=0.00001)

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
num_train_images = utils.get_num_files(TRAIN_DIR)

num_val_images = utils.get_num_files(VAL_DIR)



def lr_decay(epoch):

    if epoch%20 == 0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr/2)

        print("LR changed to {}".format(lr/2))

    return K.get_value(model.optimizer.lr)



learning_rate_schedule = LearningRateScheduler(lr_decay)



filepath="VGG16" + "_model_weights.h5"

checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')

callbacks_list = [checkpoint]



history = finetune_model.fit_generator(train_generator, epochs=100, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 

validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)