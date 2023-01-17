## Importing packages



# This R environment comes with all of CRAN and many other helpful packages preinstalled.

# You can see which packages are installed by checking out the kaggle/rstats docker image: 

# https://github.com/kaggle/docker-rstats



library(tidyverse) # metapackage with lots of helpful functions

## Running code



# In a notebook, you can run a single code cell by clicking in the cell and then hitting 

# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 

# you can run code by highlighting the code you want to run and then clicking the blue arrow

# at the bottom of this window.



## Reading in files



# You can access files from datasets you've added to this kernel in the "../input/" directory.

# You can see the files added to this kernel by running the code below. 



list.files(path = "../input")



## Saving data



# If you save any files or images, these will be put in the "output" directory. You 

# can see the output directory by committing and running your kernel (using the 

# Commit & Run button) and then checking out the compiled version of your kernel.

#LIBRERIAS 

import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense



#EXPLORACION

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

print(train_df.shape)

train_df.head()
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(test_df.shape)

test_df.head()
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    

    return im
N = train_df.shape[0]

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'

    )
N = test_df.shape[0]

x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'

    )
y_train = pd.get_dummies(train_df['diagnosis']).values



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

# Training

bs = 64 //2

path_anno = '../input/aptos2019-blindness-detection/train.csv'

path_img = '../input/aptos2019-blindness-detection/train_images'



# Test dataset

tpath_anno = '../input/aptos2019-blindness-detection/test.csv'

tpath_img = '../input/aptos2019-blindness-detection/test_images'
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))

x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=0.15, 

    random_state=2019

)
class MixupGenerator():

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):

        self.X_train = X_train

        self.y_train = y_train

        self.batch_size = batch_size

        self.alpha = alpha

        self.shuffle = shuffle

        self.sample_num = len(X_train)

        self.datagen = datagen



    def __call__(self):

        while True:

            indexes = self.__get_exploration_order()

            itr_num = int(len(indexes) // (self.batch_size * 2))



            for i in range(itr_num):

                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]

                X, y = self.__data_generation(batch_ids)



                yield X, y

                

    def __get_exploration_order(self):

        indexes = np.arange(self.sample_num)



        if self.shuffle:

            np.random.shuffle(indexes)



        return indexes



    def __data_generation(self, batch_ids):

        _, h, w, c = self.X_train.shape

        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)

        y_l = l.reshape(self.batch_size, 1)



        X1 = self.X_train[batch_ids[:self.batch_size]]

        X2 = self.X_train[batch_ids[self.batch_size:]]

        X = X1 * X_l + X2 * (1 - X_l)



        if self.datagen:

            for i in range(self.batch_size):

                X[i] = self.datagen.random_transform(X[i])

                X[i] = self.datagen.standardize(X[i])



        if isinstance(self.y_train, list):

            y = []



            for y_train_ in self.y_train:

                y1 = y_train_[batch_ids[:self.batch_size]]

                y2 = y_train_[batch_ids[self.batch_size:]]

                y.append(y1 * y_l + y2 * (1 - y_l))

        else:

            y1 = self.y_train[batch_ids[:self.batch_size]]

            y2 = self.y_train[batch_ids[self.batch_size:]]

            y = y1 * y_l + y2 * (1 - y_l)



        return X, y

BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )

data.show_batch(rows=3, figsize=(7,6))
class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_val = y_val.sum(axis=1) - 1

        

        y_pred = self.model.predict(X_val) > 0.5

        y_pred = y_pred.astype(int).sum(axis=1) - 1



        _val_kappa = cohen_kappa_score(

            y_val,

            y_pred, 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")

        

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('model.h5')



        return
densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)
def build_model():

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.00005),

        metrics=['accuracy']

    )

    

    return model
model = build_model()

model.summary()
kappa_metrics = Metrics()



history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=30,

    validation_data=(x_val, y_val),

    callbacks=[kappa_metrics]

)