# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# First, look at everything.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
# in order to import an image
from IPython.display import Image
im1 = Image("../input/walk_or_run_train/train/run/run_00061c18.png")
im1
# TRAIN

# ../input/
PATH = os.path.abspath(os.path.join('..', 'input'))

# TRAIN_RUN

# ../input/walk_or_run_train/train/run
train_run_images = os.path.join(PATH, "walk_or_run_train", "train", 'run')
# ../input/walk_or_run_train/train/run/*.png
train_run = glob(os.path.join(train_run_images, "*.png"))

# TRAIN_WALK

# ../input/walk_or_run_train/train/walk
train_walk_images = os.path.join(PATH, "walk_or_run_train", "train", 'walk')
# ../input/walk_or_run_train/train/walk/*.png
train_walk = glob(os.path.join(train_walk_images, "*.png"))

# ADD TRAIN_WALK AND TRAIN_RUN INTO A DATAFRAME

train = pd.DataFrame()
train['file'] = train_run + train_walk
train.head()
# TEST

# ../input/
PATH = os.path.abspath(os.path.join('..', 'input'))

# TEST_RUN

# ../input/walk_or_run_test/test/run
test_run_images = os.path.join(PATH, "walk_or_run_test", "test", 'run')
# ../input/walk_or_run_test/test/run/*.png
test_run = glob(os.path.join(test_run_images, "*.png"))

# TEST_WALK

# ../input/walk_or_run_test/test/walk
test_walk_images = os.path.join(PATH, "walk_or_run_test", "test", 'walk')
# ../input/walk_or_run_test/test/walk/*.png
test_walk = glob(os.path.join(test_walk_images, "*.png"))

test = pd.DataFrame()
test['file'] = test_run + test_walk
test.shape
#TRAIN LABELS

train['label'] = [1 if i in train_run else 0 for i in train['file']]
train.head()
#TEST LABELS

test['label'] = [1 if i in test_run else 0 for i in test['file']]
test.tail()
# TRAIN RUN AND WALK IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(train_run[3]))

plt.subplot(122)
plt.imshow(cv2.imread(train_walk[1]))
# TEST RUN AND WALK IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(test_run[7]))

plt.subplot(122)
plt.imshow(cv2.imread(test_walk[8]))
# create x_train3D and reshape it (coverting images into array)

x_train3D = []
for i in range(0,600):
    x_train3D.append(cv2.imread(train.file[i]).reshape(224*224,3))
    
x_train3D = np.asarray(x_train3D) # to make it array
x_train3D = x_train3D/1000 # for scaling

# create y_train
y_train = train.label
y_train = np.asarray(y_train) # to make it array
print('x_train3D shape: ', x_train3D.shape)
print('y_train shape: ', y_train.shape)
# create x_train 
# convert to gray scale = (0.299)*R + G*(0.5870) + B*(0.1140)
x_train = np.zeros((600,50176))
for i in range(0,600):
    for j in range(0,50176):
        x_train[i,j] = ((0.299*x_train3D[i][j][0])+(0.5870*x_train3D[i][j][1])+(0.1140*x_train3D[i][j][2]))

x_train = np.asarray(x_train) # to make it array
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('shape of train_run (original) image: ', cv2.imread(train_run[0]).shape)
# ORIJINAL IMAGES
# TRAIN RUN IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(train_run[8]))

plt.subplot(122)
plt.imshow(cv2.imread(train_run[11]))
print('shape of x_train (modified) image: ', x_train[0].reshape(224,224).shape)
# MODIFIED ONES

img_size = 224
plt.figure(figsize=(16,16))
plt.subplot(1, 2, 1)
plt.imshow(x_train[8].reshape(img_size, img_size))
plt.subplot(1, 2, 2)
plt.imshow(x_train[11].reshape(img_size, img_size))
# create x_test and reshape it (coverting images into array)
x_test3D = []
for i in range(0,141):
    x_test3D.append(cv2.imread(test.file[i]).reshape(224*224,3))

x_test3D = np.asarray(x_test3D) # to make it array

# create y_test
y_test = test.label
y_test = np.asarray(y_test) # to make it array
print('x_test3D shape: ', x_test3D.shape)
print('y_test shape: ', y_test.shape)
# create x_test 
# convert to gray scale = (0.299)*R + G*(0.5870) + B*(0.1140)
x_test = np.zeros((141,50176))
for i in range(0,141):
    for j in range(0,50176):
        x_test[i,j] = ((0.299*x_test3D[i][j][0])+(0.5870*x_test3D[i][j][1])+(0.1140*x_test3D[i][j][2]))

x_test = np.asarray(x_test) # to make it array
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)
# Reshape (to be suitable for keras libarary)
x_train = x_train.reshape(-1,224,224,1)
x_test = x_test.reshape(-1,224,224,1)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 18, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (224,224,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.10))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(10, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(6, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))
# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])
epochs = 100  # for better result increase the epochs
batch_size = 3
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.1,  # randomly rotate images in the range 1 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test), steps_per_epoch=x_train.shape[0] // batch_size)
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
