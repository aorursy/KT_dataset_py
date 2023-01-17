# Importing all the required libraries

import numpy as np 
import pandas as pd 
import os
import random
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline

# Checking the directory 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Visually Inspect Image Dataset 

input_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human'

fig, ax = plt.subplots(2, 2, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()

for i, _set in enumerate(['train', 'validation']):
    set_path = input_path+'/'+_set
    ax[i].imshow(plt.imread(set_path+'/horses/'+os.listdir(set_path+'/horses')[0]), cmap='gray')
    ax[i].set_title('Set: {}, type:horses'.format(_set))
    ax[i+2].imshow(plt.imread(set_path+'/humans/'+os.listdir(set_path+'/humans')[0]), cmap='gray')
    ax[i+2].set_title('Set: {}, type:humans'.format(_set))
    
# Image Preprocessing

input_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human'

def process_data(img_dims, batch_size):
  
   
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    # This is fed to the network in the specified batch sizes and image dimensions
    train_gen = train_datagen.flow_from_directory(
    directory=input_path + '/train/', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)

    test_gen = test_val_datagen.flow_from_directory(
    directory=input_path + '/validation/', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)
    

    return train_gen, test_gen
# Hyperparameters

img_dims = 200
epochs = 10
batch_size = 20

# Getting the data
train_gen, test_gen = process_data(img_dims, batch_size)
#Inception V3 
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(200,200,3),include_top=False,weights='imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()
# Fully Connected Layer
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
Dl_1 = tf.keras.layers.Dropout(rate = 0.2)
pre_prediction_layer = tf.keras.layers.Dense(180, activation='relu')
Dl_2 = tf.keras.layers.Dropout(rate = 0.2)
prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')


model_V3 = tf.keras.Sequential([
  pre_trained_model,
  global_average_layer,
  Dl_1,
  pre_prediction_layer,
  Dl_2,
  prediction_layer
])
#Compiling Fully Connected Layer
model_V3.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_V3.summary()

# I will be using the following to reduce the learning rate by the factor of 0.2 when the 'val_loss' will increase in consecutive 3 epochs.
# Callbacks 
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=2, mode='max')
#Fitting the model
hist = model_V3.fit_generator(
           train_gen, steps_per_epoch=50, 
           epochs=10, validation_data=test_gen, 
           validation_steps=12 , callbacks=[lr_reduce])
