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





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')



df.head(5)
def preprocess_image(image_path, desired_size=96):

    im = cv2.imread(image_path)

    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    im = cv2.resize(im, (desired_size, desired_size))

    im = cv2.addWeighted(im, 4, cv2.blur(im, ksize=(10,10)), -4, 128)

    return im
N = df.shape[0]

X = np.empty((N, 96, 96, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(df['id_code'])):

    X[i, :, :, :] = preprocess_image(

        f'../input/train_images/{image_id}.png'

    )
y = df['diagnosis']

y.head(10)
from keras.utils import np_utils

y1 = np_utils.to_categorical(y)

print('The shape of y1 is:', y1.shape)

num_classes = y1.shape[1]

y1[:10]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)

print('The shape of X_train is:', X_train.shape)

print('The shape of y_train is:', y_train.shape)

print('The shape of X_test is:', X_test.shape)

print('The shape of y_test is:', y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, GaussianNoise, GaussianDropout

from keras.layers import Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, SeparableConv2D



from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

from keras import regularizers



in_size = (96,96,3)
# Import necessary packages

import argparse



# Import necessary components to build LeNet

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2



def alexnet_model(img_shape=in_size, n_classes=num_classes, l2_reg=0.,

    weights=None):



    # Initialize model

    alexnet = Sequential()



    # Layer 1

    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,

        padding='same', kernel_regularizer=l2(l2_reg)))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(MaxPooling2D(pool_size=(2, 2)))



    # Layer 2

    alexnet.add(Conv2D(256, (5, 5), padding='same'))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(MaxPooling2D(pool_size=(2, 2)))



    # Layer 3

    alexnet.add(ZeroPadding2D((1, 1)))

    alexnet.add(Conv2D(512, (3, 3), padding='same'))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(MaxPooling2D(pool_size=(2, 2)))



    # Layer 4

    alexnet.add(ZeroPadding2D((1, 1)))

    alexnet.add(Conv2D(1024, (3, 3), padding='same'))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))



    # Layer 5

    alexnet.add(ZeroPadding2D((1, 1)))

    alexnet.add(Conv2D(1024, (3, 3), padding='same'))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(MaxPooling2D(pool_size=(2, 2)))



    # Layer 6

    alexnet.add(Flatten())

    alexnet.add(Dense(3072))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(Dropout(0.5))



    # Layer 7

    alexnet.add(Dense(4096))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('relu'))

    alexnet.add(Dropout(0.5))



    # Layer 8

    alexnet.add(Dense(n_classes))

    alexnet.add(BatchNormalization())

    alexnet.add(Activation('softmax'))



    if weights is not None:

        alexnet.load_weights(weights)

    return alexnet
model = alexnet_model()

model.summary()
from sklearn.metrics import cohen_kappa_score, accuracy_score

from keras.callbacks import Callback, ModelCheckpoint



class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_pred = self.model.predict(X_val)



        _val_kappa = cohen_kappa_score(

            y_val.argmax(axis=1), 

            y_pred.argmax(axis=1), 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")



        return
from keras.optimizers import Adam



model.compile(loss='categorical_crossentropy', 

              optimizer=Adam(lr=0.00005), metrics=['accuracy'])
BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.10,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(X_train, y_train, batch_size=BATCH_SIZE)
kappa_metrics = Metrics()



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    data_generator,

    steps_per_epoch=X_train.shape[0] / BATCH_SIZE,

    epochs=50,

    validation_data=(X_test, y_test),

    callbacks=[checkpoint, kappa_metrics]

)
y_pred = model.predict(X_train)
from matplotlib import pyplot as plt



fig = plt.plot(history.history["acc"],label = "train", color='green')

plt.plot(history.history["val_acc"],label = "test", color='red')

plt.legend(loc='upper left')

plt.xlabel("epochs")

plt.ylabel("accuracy")

plt.title("accuracy by epochs")

plt.show()
test_df = pd.read_csv('../input/sample_submission.csv')
N = test_df.shape[0]

x_test = np.empty((N, 96, 96, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'../input/test_images/{image_id}.png'

    )
model.load_weights('model.h5')

y_test = model.predict(x_test, verbose=2)



test_df['diagnosis'] = y_test.argmax(axis=1)

print(test_df.head(10))



test_df.to_csv('submission.csv',index=False)