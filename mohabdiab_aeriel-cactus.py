# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
os.listdir('../input/')
train = pd.read_csv('../input/train.csv')
train.head()
training_path = '../input/train/train/'

test_path = '../input/test/test/'
from tqdm import tqdm, tqdm_notebook

import cv2 as cv





images_train = []

labels_train = []



images = train['id'].values

for image_id in tqdm_notebook(images):

    

    image = np.array(cv.imread(training_path + image_id))

    label = train[train['id'] == image_id]['has_cactus'].values[0]

    

    images_train.append(image)

    labels_train.append(label)

    

    images_train.append(np.flip(image))

    labels_train.append(label)

    

    images_train.append(np.flipud(image))

    labels_train.append(label)

    

    images_train.append(np.fliplr(image))

    labels_train.append(label)

    

    

images_train = np.asarray(images_train)

images_train = images_train.astype('float32')

images_train /= 255.



labels_train = np.asarray(labels_train)
test_images_names = []



for filename in os.listdir(test_path):

    test_images_names.append(filename)

    

test_images_names.sort()



images_test = []



for image_id in tqdm_notebook(test_images_names):

    images_test.append(np.array(cv.imread(test_path + image_id)))

    

images_test = np.asarray(images_test)

images_test = images_test.astype('float32')

images_test /= 255
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(images_train, labels_train, test_size = 0.2, stratify = labels_train)
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, DepthwiseConv2D, Flatten

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping





model = Sequential()

        

model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))

    

model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

    

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

    

model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))

model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

    

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))    

    

model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

    

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

    

#model.add(GlobalAveragePooling2D())

model.add(Flatten())

    

model.add(Dense(470, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

    

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

    

model.add(Dense(128, activation = 'tanh'))



    

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])

    

file_path = 'weights-aerial-cactus.h5'



callbacks = [

        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),

        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, mode = 'min', min_lr = 0.00001),

        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 15, verbose = 1, restore_best_weights = True)

        ]

here_we_go = model.fit(X_train, 

            y_train, 

            batch_size = 128, 

            epochs = 80, 

            validation_data = (X_test, y_test),

            verbose = 1,

            callbacks = callbacks)
test_df = pd.read_csv('../input/sample_submission.csv')

test = []

images_test = test_df['id'].values



for img_id in tqdm_notebook(images_test):

    test.append(cv.imread(test_path + img_id))

    

test = np.asarray(test)

test = test.astype('float32')

test /= 255



y_test_pred = model.predict_proba(test)



test_df['has_cactus'] = y_test_pred



test_df.to_csv('aerial-cactus-submission.csv', index = False)


