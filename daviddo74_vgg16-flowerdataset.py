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

NUM_CLASSES = 17

IMG_SIZE = 224
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

DATA_DIR_TRAIN = "../input/dataset/dataset"

CATEGORIES = os.listdir(DATA_DIR_TRAIN)

x_train = []

y_train = []

for category in CATEGORIES:

    print("Now is loading class: ", category, "...")

    path = os.path.join(DATA_DIR_TRAIN, category)   

    class_num = CATEGORIES.index(category)

    for p in os.listdir(path):

        img_path = os.path.join(path, p)

        img = cv2.imread(img_path)

        new_img = preprocess_img(img)

        x_train.append(new_img)

        y_train.append(class_num)
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

  plt.title(CATEGORIES[y_train[random_array[i]]])

  plt.xticks([])

  plt.yticks([])
import keras

from sklearn.model_selection import train_test_split
base_model=keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=(3,IMG_SIZE,IMG_SIZE), pooling='avg')
base_model.summary()
X_train, X_val, Y_train, Y_val = train_test_split(x_train_np, y_train_np, test_size=0.25, random_state=42)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
epochs=50

batch_size=128

red_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
model=Sequential()

model.add(base_model)



model.add(Dense(256,activation='relu'))

model.add(keras.layers.BatchNormalization())

model.add(Dense(NUM_CLASSES,activation='softmax'))





for layer in base_model.layers:

    layer.trainable=True



model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
History = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = 50, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)