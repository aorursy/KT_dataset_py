# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage import io, color, exposure, transform

import os

import glob

import h5py



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, model_from_json

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D



from keras.optimizers import SGD

from keras.utils import np_utils

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras import backend as K

K.set_image_data_format('channels_first')



from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

NUM_CLASSES = 43

IMG_SIZE = 48
def preprocess_img(img):

    # Histogram normalization in y

    hsv = color.rgb2hsv(img)

    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])

    img = color.hsv2rgb(hsv)



    # central scrop

    min_side = min(img.shape[:-1])

    centre = img.shape[0]//2, img.shape[1]//2

    img = img[centre[0]-min_side//2:centre[0]+min_side//2,

              centre[1]-min_side//2:centre[1]+min_side//2,

              :]



    # rescale to standard size

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))



    # roll color axis to axis 0

    img = np.rollaxis(img,-1)



    return img
import cv2

DATA_DIR_TRAIN = "../input/gtsrb_challenge/GTSRB_Challenge/train"

x_train = []

y_train = []

for i in range(43):

    category = ""

    if i<10:

        category = "0000"+str(i)

    else:

        category = "000"+str(i)

    path = os.path.join(DATA_DIR_TRAIN, category)   

    

    for p in os.listdir(path):

        img_path = os.path.join(path, p)

        img = cv2.imread(img_path)

        new_img = preprocess_img(img)

        x_train.append(new_img)

        y_train.append(i)

x_train_np = np.array(x_train, dtype='float32')         

y_train_np = np.eye(NUM_CLASSES, dtype='uint8')[y_train]



print(x_train_np.shape)

print(y_train_np.shape)
random_array = np.random.randint(len(x_train),size=100)

random_array
grids = (10,10)

counter = 0



plt.figure(figsize=(20,20))



for i in range(0, 100):

  ax = plt.subplot(10, 10, i+1)

  img = np.rollaxis(x_train[random_array[i]], 0, 3)

  ax = plt.imshow(img, cmap='gray')

  plt.title(y_train[random_array[i]])

  plt.xticks([])

  plt.yticks([])
random_array = np.random.randint(len(x_train),size=100)

random_array
grids = (10,10)

counter = 0



plt.figure(figsize=(20,20))



for i in range(0, 100):

  ax = plt.subplot(10, 10, i+1)

  img = np.rollaxis(x_train[random_array[i]], 0, 3)

  ax = plt.imshow(img, cmap='gray')

  plt.title(y_train[random_array[i]])

  plt.xticks([])

  plt.yticks([])
def cnn_model():

    model = Sequential()



    model.add(Conv2D(32, (3, 3), padding='same',

                     input_shape=(3, IMG_SIZE, IMG_SIZE),

                     activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), padding='same',

                     activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, (3, 3), padding='same',

                     activation='relu'))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model



def lr_schedule(epoch):

    return lr*(0.1**int(epoch/10))
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train_np, y_train_np, test_size=0.2, random_state=42)



datagen = ImageDataGenerator(featurewise_center=False, 

                            featurewise_std_normalization=False, 

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            zoom_range=0.2,

                            shear_range=0.1,

                            rotation_range=10.,)



datagen.fit(X_train)
model = cnn_model()

lr = 0.01

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

          optimizer=sgd,

          metrics=['accuracy'])





def lr_schedule(epoch):

    return lr*(0.1**int(epoch/10))
batch_size = 32

nb_epoch = 1

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                            steps_per_epoch=X_train.shape[0],

                            epochs=nb_epoch,

                            validation_data=(X_val, Y_val),

                            callbacks=[LearningRateScheduler(lr_schedule),

                                       ModelCheckpoint('model.h5',save_best_only=True)]

                           )
DATA_DIR_TEST = "../input/gtsrb_challenge/GTSRB_Challenge/test"

x_test = []

for p in os.listdir(DATA_DIR_TEST):

    img_path = os.path.join(DATA_DIR_TEST, p)

    img = cv2.imread(img_path)

    new_img = preprocess_img(img)

    x_test.append(new_img)
random_array = np.random.randint(len(x_test),size=100)

random_array
grids = (10,10)

counter = 0



plt.figure(figsize=(20,20))



for i in range(0, 100):

    ax = plt.subplot(10, 10, i+1)

    img = np.rollaxis(x_test[random_array[i]], 0, 3)

    ax = plt.imshow(img, cmap='gray')

    x = x_test[random_array[i]]

    y_predict = np.argmax(model.predict(x.reshape(1,3,48,48)), axis=1)

    plt.title(y_predict)

    plt.xticks([])

    plt.yticks([])
import csv

with open('dungdmse05228_submission.csv', mode='w') as csv_file:

    fieldnames = ['Filename', 'ClassId']

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)



    writer.writeheader()

    DATA_DIR_TEST = "../input/gtsrb_challenge/GTSRB_Challenge/test"

    

    for p in os.listdir(DATA_DIR_TEST):

        img_path = os.path.join(DATA_DIR_TEST, p)

        img = cv2.imread(img_path)

        new_img = preprocess_img(img)

        y_predict = np.argmax(model.predict(new_img.reshape(1,3,48,48)), axis=1)

        

        writer.writerow({'Filename': p, 'ClassId': int(y_predict)})
img = cv2.imread("../input/gtsrb_challenge/GTSRB_Challenge/train/00009/00000_00000.ppm")

img = cv2.resize(img, (120, 120))

plt.imshow(img)