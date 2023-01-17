import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Utilities
import sys
import h5py
import warnings
from time import time
import os
from os import path
from zipfile import ZipFile
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import random
from scipy import ndarray

import skimage as sk
from skimage import transform
from skimage import util
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# KERAS MODULES
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import keras
import tensorflow as tf 

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
project_path = "/kaggle/input/dog-breed-identification"
import os
os.chdir(project_path)
labels = pd.read_csv("labels.csv")
labels.head()
labels.shape
breed_count = labels['breed'].value_counts()
breed_count
targets_labels = pd.Series(labels['breed'])
one_hot = pd.get_dummies(targets_labels,sparse=True)
one_hot_labels = np.asarray(one_hot)
one_hot_labels[0]
img_rows = 128
img_cols = 128
x_features = []
y_features = []
from tqdm import tqdm
for f,img_label in tqdm(labels.values): # f for format ,jpg
    img = cv2.imread('./train/{}.jpg'.format(f), 1)
    img_resize = cv2.resize(img, (img_rows, img_cols)) 
    x_features.append(img_resize)
    y_features.append(img_label)
plt.imshow(x_features[0])
y_features[0]
x_train_data = np.asarray(x_features,np.float32) / 255
print(x_train_data.shape)
#y_train.shape

label_encoder = preprocessing.LabelEncoder()
encodedlabels = label_encoder.fit_transform(y_features)
print('Classes'+str(label_encoder.classes_))
y_train_encoded = np_utils.to_categorical(encodedlabels)
classes = y_train_encoded.shape[1]
print(str(classes))
#Split into train/validation set

x_train,x_val,y_train,y_val = train_test_split(x_train_data,y_train_encoded,test_size=0.2,random_state=17)
submission = pd.read_csv("sample_submission.csv")
test_img = submission['id']
submission.head()
x_test_feature = []
i = 0 # initialisation
for f in tqdm(test_img.values): # f for format ,jpg
    img = cv2.imread('./test/{}.jpg'.format(f), 1)
    img_resize = cv2.resize(img, (img_rows, img_cols)) 
    x_test_feature.append(img_resize)
x_test_data = np.asarray(x_test_feature,np.float32) / 255
x_train_data[0].shape
class_count = breed_count.size
IMAGE_DIMS = x_train_data[0].shape
base_model = Sequential()
base_model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=IMAGE_DIMS, activation='relu'))
base_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
base_model.add(BatchNormalization(axis=3))
base_model.add(Flatten())
# Dense layer with 256 neurons with relu activation
base_model.add(Dense(256, activation='relu'))
# Dense layer with 120 neurons as final layer 
base_model.add(Dense(class_count, activation='softmax'))
base_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
base_model.summary()
print(y_train[0])
plt.imshow(x_train[0])
EPOCHS=10
BATCH_SIZE=125
base_model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data=(x_val, y_val))
model_augmented = Sequential()

model_augmented.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=IMAGE_DIMS, activation='relu'))
model_augmented.add(BatchNormalization(axis=3))
model_augmented.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model_augmented.add(MaxPooling2D((2, 2)))
model_augmented.add(BatchNormalization(axis=3))
model_augmented.add(Dropout(0.1))

model_augmented.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model_augmented.add(BatchNormalization(axis=3))
model_augmented.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model_augmented.add(MaxPooling2D((2, 2)))
model_augmented.add(BatchNormalization(axis=3))
model_augmented.add(Dropout(0.1))

model_augmented.add(Flatten())
model_augmented.add(Dense(256, activation='relu'))
model_augmented.add(BatchNormalization())
model_augmented.add(Dropout(0.5))

model_augmented.add(Dense(classes, activation='softmax'))

model_augmented.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_augmented.summary()
# SETTING UP CHECKPOINTS, CALLBACKS AND REDUCING LEARNING RATE
lrr = ReduceLROnPlateau(monitor='val_acc', 
                        patience=3, 
                        verbose=2, 
                        factor=0.4, 
                        min_lr=0.00001)

filepath="weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 
                              verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoints, lrr]
train_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
                     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                     horizontal_flip=True, fill_mode="nearest")
train_generator.fit(x_train)
val_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode="nearest")
val_generator.fit(x_val)
model_augmented.fit_generator(train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(x_train) / BATCH_SIZE, 
                    epochs=EPOCHS, validation_data=val_generator.flow(x_val, y_val, batch_size=BATCH_SIZE), callbacks=callbacks_list)
from keras.applications.vgg16 import VGG16, preprocess_input
# Instantiate the model with the pre-trained weights (no top)
base_model_VGG16= VGG16(weights=('/content/drive/My Drive/greatlakes/Projects/CNN/Project2/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                 include_top=False, pooling='avg')
base_model_VGG16.summary()
from keras.layers import Input
input = Input(shape=(128,128,3),name = 'image_input')
output_vgg16 = base_model_VGG16(input)
transfer_learn_model = Dense(1024, activation='relu')(output_vgg16)
transfer_learn_model = BatchNormalization()(transfer_learn_model)
transfer_learn_model = Dense(256, activation='relu')(transfer_learn_model)
transfer_learn_model=  Dropout(0.3)(transfer_learn_model)
transfer_learn_model = Dense(120, activation='softmax', name='predictions',kernel_initializer='uniform')(transfer_learn_model)
VGG16_transfer_learn_model = Model(input=input, output=transfer_learn_model)
#Freezing layers in the model which don't have 'dense' in their name
for layer in base_model_VGG16.layers:
    #Freezing a layer
    layer.trainable = False

#Module to print colourful statements
from termcolor import colored

#Check which layers have been frozen 
for layer in VGG16_transfer_learn_model.layers:
  print (colored(layer.name, 'blue'))
  print (colored(layer.trainable, 'red'))

VGG16_transfer_learn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
VGG16_transfer_learn_model.summary()
EPOCHS=10
BATCH_SIZE=128
VGG16_transfer_learn_model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data=(x_val, y_val), callbacks=callbacks_list)
