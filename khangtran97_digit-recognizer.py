import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 128

NUM_CLASSES = 10





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
import os

os.listdir('../input/')
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
train.shape
test.head()
test.shape
y = train["label"]



# Drop 'label' column

x = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(y)



y.value_counts()
# Normalize the data

x = x / 255.0

test = test / 255.0
x = x.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y = to_categorical(y, num_classes = 10)
# Split the train and the validation set for the fitting

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1, random_state=2, stratify = y, shuffle = True)
# Some examples

g = plt.imshow(x_train[0][:,:,0])
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
x_train[0].shape
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
from keras.optimizers import RMSprop

# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau)



epochs = 80; batch_size = 1024

checkpoint = ModelCheckpoint('../working/Resnet50-visible.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)

callbacks_list = [checkpoint, reduceLROnPlat, early]
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# With data augmentation to prevent overfitting (accuracy 0.99286)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
batch_size = 1024

# Fit the model

epochs = 80

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_valid,y_valid),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=callbacks_list)
model.load_weights('../working/Resnet50-visible.h5')

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)