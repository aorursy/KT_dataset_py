import re

import gc

import os

import cv2

import glob

import keras

import shutil

import pathlib

import PIL

import numpy as np

import pandas as pd

import seaborn as sb

import networkx as nx

import tensorflow as tf

import matplotlib.pyplot as plt

from shutil import copyfile

import tensorboard

from datetime import datetime

from packaging import version

from tensorflow import keras as ks

from tensorflow.keras import datasets, layers, models

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

from skimage.io import imread, imshow, imsave, imread_collection

from mpl_toolkits.mplot3d import Axes3D

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing.image import img_to_array, array_to_img

from keras_preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from tqdm import tqdm
# Show versions

print('TensorFlow Version: {}'.format(tf.__version__))

print('Eager execution: {}'.format(tf.executing_eagerly()))

print('OpenCV Version:{}'.format(cv2.__version__))

print('Keras Version:{}'.format(ks.__version__))

print('Numpy Version:{}'.format(np.__version__))

print('Pandas Version:{}'.format(pd.__version__))
# Settings

epochs = 10

img_height = 312

img_width = 312

batch_size = 64
# Check if GPU is ready

print(tf.test.is_gpu_available())
# Check the number of GPU's that are ready

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Read in CSV

train=pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")

print(train)
train.shape
train.describe()
# Read in CSV

test=pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

print(test)
# Read in CSV

sample_submission="../input/plant-pathology-2020-fgvc7/sample_submission.csv"

submission=pd.read_csv(sample_submission)
# Create training folder 

pathto="./Train/"

shutil.os.mkdir(pathto)
# Create testing folder

pathto="./Test/"

shutil.os.mkdir(pathto)
# Split images based on name

for path in glob.iglob(r'../input/plant-pathology-2020-fgvc7/images/*.jpg'):

    match = re.search(r'\bTest_',path)

    if match:

        shutil.copy(path, "./Test")

        print("Sent to test folder -",path)

    else:

        shutil.copy(path, "./Train")

        print("Sent to train folder -",path)
# Grab a sample image

for dirname, _, filenames in os.walk('/kaggle/input/plant-pathology-2020-fgvc7/images/'):

    for filename in filenames[1:2]:

        picture = imread('../input/plant-pathology-2020-fgvc7/images/'+filename)

        plt.figure(figsize=(20, 15))

        plt.title(filename)

        plt.grid()

        plt.ylabel('Height {}'.format(picture.shape[0]))

        plt.xlabel('Width {}'.format(picture.shape[1]))

        plt.imshow(picture);
# View all images

col_dir = '../input/plant-pathology-2020-fgvc7/images/*.jpg'

images = imread_collection(col_dir)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20,20))



for i, image in enumerate(images):

    if (i == 25) : break

    row = i // 5

    col = i % 5

    axes[row, col].axis("off")

    axes[row, col].imshow(image, aspect="auto")

plt.subplots_adjust(wspace=.05, hspace=.05)
# View all training images

col_dir = './Train/*.jpg'

images = imread_collection(col_dir)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20,20))



for i, image in enumerate(images):

    if (i == 25) : break

    row = i // 5

    col = i % 5

    axes[row, col].imshow(image, aspect="auto")

plt.subplots_adjust(wspace=0.3, hspace=0.3)
# View all testing images

col_dir = './Test/*.jpg'

images = imread_collection(col_dir)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20,20))



for i, image in enumerate(images):

    if (i == 25) : break

    row = i // 5

    col = i % 5

    axes[row, col].imshow(image, aspect="auto")

plt.subplots_adjust(wspace=0.3, hspace=0.3)
# View all images as negatives

col_dir = '../input/plant-pathology-2020-fgvc7/images/*.jpg'

images = imread_collection(col_dir)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20,20))



for i, image in enumerate(images):

    if (i == 25) : break

    row = i // 5

    col = i % 5

    negative = 255 - image

    axes[row, col].imshow(negative, aspect="auto")

plt.subplots_adjust(wspace=0.3, hspace=0.3)
next(train.iterrows())[1]
# Row data in pandas are returned as a series

for index, row in train.head(n=5).iterrows():

    print(index,row)
# Row data in pandas are returned as a series

for index2, row2 in test.head(n=5).iterrows():

    print(index2,row2)
# Append filetype to each image

for i in tqdm(range(train.shape[0])):

    img=(train['image_id'])

add = img.astype(str)+".jpg"

train['image_id'] = add 
image_id=add

image_id.unique()
# Append filetype to each image

for i in tqdm(range(test.shape[0])):

    img=(test['image_id'][:])

add = img.astype(str)+".jpg"

test['image_id'] = add
image_id=test['image_id']

image_id.unique()
# Classification labels

column_names=["healthy","multiple_diseases","rust","scab"]



# Assign each image a condition

healthy=(1,0,0,0)

multiple_diseases = (0,1,0,0)

rust=(0,0,1,0)

scab=(0,0,0,1)



# Count the amount of conditions

health_count=0

md_count=0

rust_count=0

scab_count=0



for index, row in train.iterrows():

    condition=index, row['image_id'],row['healthy'],row['multiple_diseases'],row['rust'],row['scab']

    if condition[2:6]==healthy:

        health_count+=1

        print(condition[1]+"- This leaf is healthy:",healthy)

    if condition[2:6]==multiple_diseases:

        md_count+=1

        print(condition[1]+"- This leaf has multiple diseases:",multiple_diseases)

    if condition[2:6]==rust:

        rust_count+=1

        print(condition[1]+"- This leaf has rust:",rust)       

    if condition[2:6]==scab:

        scab_count+=1

        print(condition[1]+"- This leaf has a scab:",scab)
# Display the amount of conditions for each category

print("The amount of healthy leaves:",health_count)

print("The amount of multiple diseased leaves:",md_count)

print("The amount of rust leaves:",rust_count)

print("The amount of scab leaves:",scab_count)
# Display the total amount

amount = health_count+md_count+rust_count+scab_count

print("Total amount of conditions:",amount)
# ImageGenerator

# https://keras.io/preprocessing/image/

datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=45,

      zca_whitening=False,

      zca_epsilon=1e-06,

      width_shift_range=0.2,

      height_shift_range=0.2,

      brightness_range=(0.1,1.0),

      channel_shift_range=5.0,

      shear_range=0.2,

      validation_split=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      vertical_flip=True,

      fill_mode='nearest')



test_datagen=ImageDataGenerator(rescale=1./255)



train_image_generator=datagen.flow_from_dataframe(

    dataframe=train[:1460],

    directory='../input/plant-pathology-2020-fgvc7/images/',

    x_col="image_id",

    y_col=column_names,

    batch_size=batch_size,

    seed=42,

    shuffle=True,

    class_mode="raw",

    target_size=(img_height,img_width))



valid_image_generator=test_datagen.flow_from_dataframe(

    dataframe=train[1460:],

    directory='../input/plant-pathology-2020-fgvc7/images/',

    x_col="image_id",

    y_col=column_names,

    batch_size=batch_size,

    seed=42,

    shuffle=True,

    class_mode="raw",

    target_size=(img_height,img_width))



test_image_generator=test_datagen.flow_from_dataframe(

    dataframe=test[:],

    directory='../input/plant-pathology-2020-fgvc7/images/',

    x_col="image_id",

    batch_size=1,

    seed=42,

    shuffle=False,

    class_mode=None,

    target_size=(img_height,img_width))
# Apply two dimensional convolutional layer over images that is convolved with the current layer to produce tensor outputs

model=keras.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(img_height,img_width,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))



# Compile the model

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# Display model

model.summary()
%%time

history=model.fit_generator(generator=train_image_generator,

                            validation_data=valid_image_generator,

                            epochs=epochs)
# https://www.tensorflow.org/tutorials/images/classification?hl=da

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
# Reset CNN

test_image_generator.reset()

prediction=model.predict_generator(test_image_generator,verbose=1)



# Submit Submission

submission.loc[:, 'healthy':] = prediction

submission.to_csv('submission.csv', index=False)

submission.head()