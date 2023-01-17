# Pnuemonia Detection from Xrays using VGG16 petrained model
import os

import h5py

import imgaug as aug

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mimg

import imgaug.augmenters as augment

import tensorflow as tf

from keras.models import Sequential, Model

from keras.applications.xception import Xception

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D, GlobalAveragePooling2D

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras import backend as K

from keras.optimizers import Adam, SGD, RMSprop

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,EarlyStopping,TensorBoard,ReduceLROnPlateau,CSVLogger,LearningRateScheduler

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

color = sns.color_palette()

%matplotlib inline

# Implement Learning rate decay

def step_decay(epoch):

    initial_rate = 0.1

    drop = 0.5

    epochs_drop = 5.0

    learning_rate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drops))

    return learning_rate
train_data = '../input/chest_xray/chest_xray/train/'

val_data = '../input/chest_xray/chest_xray/val/'

test_data = '../input/chest_xray/chest_xray/test/'



normal_data_dir = '../input/chest_xray/chest_xray/train/NORMAL/'

pneumonia_data_dir = '../input/chest_xray/chest_xray/train/PNEUMONIA/'
aug_gen = ImageDataGenerator(

rescale = 1./255,

shear_range = 0.2,

zoom_range = 0.2,

horizontal_flip = True)



val_augs = ImageDataGenerator(rescale =  1./255)



train_gen = aug_gen.flow_from_directory(

train_data, 

target_size = (224, 224),

batch_size = 16,

shuffle = True,

class_mode = 'binary')



val_gen = val_augs.flow_from_directory(

val_data, 

target_size = (224, 224), 

batch_size = 16,

shuffle = True,

class_mode = 'binary')



test_gen = val_augs.flow_from_directory(

test_data, 

target_size = (224, 224), 

batch_size = 8,

shuffle = True,

class_mode = 'binary')
base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))



for layer in base_model.layers:

    layer.trainable = False



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

out = Dense(1, activation='sigmoid')(x)

classifier = Model(base_model.input, out)



classifier.summary()
checkpoint = ModelCheckpoint('./base.model',

                            monitor = 'val_loss',

                            verbose = 1,

                            save_best_only = True,

                            mode = 'max',

                            save_weights_only = False,

                            period = 1)



earlystop = EarlyStopping(monitor = 'val_loss',

                         min_delta = 0.001,

                         patience = 30,

                         verbose = 1,

                         mode = 'auto')



tensorboard = TensorBoard(log_dir = './logs',

                         histogram_freq = 0,

                         batch_size = 16,

                         write_graph = True,

                         write_grads = True,

                         write_images = False)



csvlogger = CSVLogger(filename = 'training_csv.log',

                     separator = ",",

                     append = False)



lrsched = LearningRateScheduler(step_decay, verbose = 1)



reduce = ReduceLROnPlateau(monitor = 'val_loss',

                          factor = 0.8,

                          patience = 5,

                          verbose = 1,

                          mode = 'auto',

                          min_delta = 0.0001,

                          cooldown = 1,

                          min_lr = 0.0001)



callbacks = [checkpoint, tensorboard, earlystop, csvlogger, reduce]

classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



history = classifier.fit_generator(train_gen,

                                  epochs = 80,

                                  steps_per_epoch = 30,

                                  validation_data = test_gen,

                                  validation_steps = 30,

                                  callbacks = callbacks,

                                  verbose = 1)
classifier_eval = classifier.evaluate_generator(test_gen)

print('classifier: Evaluation: Test Loss', classifier_eval[0])

print('classifier: Test Accuracy', classifier_eval[1])

classifier_json = classifier.to_json()

with open('classifier.json', 'w') as file:

    file.write(classifier_json)



classifier.save('classifier.h5')