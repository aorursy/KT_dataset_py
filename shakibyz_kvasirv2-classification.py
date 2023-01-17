# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/kvasir-dataset-v2/"))



# Any results you write to the current directory are saved as output.
import cv2

import glob

import keras

from keras.utils import np_utils

from keras.models import Sequential , Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.optimizers import SGD, Adam, RMSprop, Nadam



from keras.utils import to_categorical

from matplotlib import pyplot

from keras.layers import  Dropout, BatchNormalization , Activation , Input

from keras.regularizers import l1,l2

from keras.applications.mobilenet_v2 import MobileNetV2

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

import cv2

import matplotlib.pyplot as plt
path = "../input/kvasir-dataset-v2/kvasir-dataset-v2/"

categories = next(os.walk(path))[1]

X , Y = [] , []

count = 0

for category in categories:

    ids = next(os.walk(path + category))[2]

    for i in ids:

        img = cv2.imread(path + category + '/' + i)

        img_resized = cv2.resize(img,(128,128), interpolation=cv2.INTER_AREA)

        img_rsd_normalized = img_resized / 255.0

        X.append(img_rsd_normalized)

        Y.append(count)

    count += 1 

print(categories)
X , Y = np.array(X) , np.array(Y)

Y = np_utils.to_categorical(Y)

print(X.shape , Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)



X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=5)



print(X_train.shape , X_val.shape , X_test.shape)
from keras.preprocessing.image import ImageDataGenerator



BATCH_SIZE = 16

TRAIN_DIR = '../input/kvasir-dataset-v2/kvasir-dataset-v2'



train_datagen = ImageDataGenerator(

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=10,

        width_shift_range=0.1,  

        height_shift_range=0.1,

        horizontal_flip=True)



val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()



train_datagen.fit(X_train)

val_datagen.fit(X_val)

test_datagen.fit(X_test)

def create_model(input_shape, n_out):

    

    pretrain_model = VGG19(

        include_top=False, 

        weights=None, 

        input_shape=input_shape)    

    

    input_tensor = Input(shape=input_shape)

    bn = BatchNormalization()(input_tensor)

    x = pretrain_model(bn)

    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    output = Dense(n_out, activation='softmax')(x)

    model = Model(input_tensor, output)

    

    return model

model = create_model((128,128,3),8)

model.compile(optimizer=Adam(lr=2e-5,decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()




# Defining some Callbacks

import math

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import numpy as np







weight_path="{}_weights_mymodel_vgg19_AdamOpt.hdf5".format('mykaggle')



# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

#                              save_best_only=True, mode='min', save_weights_only = True)



checkpoint = ModelCheckpoint(weight_path, verbose=2, monitor='val_loss',save_best_only=True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=7, verbose=1, mode='min', min_delta=0.0001, cooldown=5, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss",  mode="min",   patience=12) 

callbacks_list = [reduceLROnPlat,early,checkpoint]





history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=(X_val,y_val),

                    steps_per_epoch=len(X_train) / BATCH_SIZE, epochs=50 , callbacks=callbacks_list)



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(model.history.history["loss"], label="loss")

plt.plot(model.history.history["val_loss"], label="val_loss")

plt.plot( np.argmin(model.history.history["val_loss"]), np.min(model.history.history["val_loss"]), marker="x", color="r", label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend();
#print(model.evaluate_generator(valid_generator,STEP_SIZE_VALID))

# num_valid_steps = ((num_dataset_samples*VALIDATION_SPLIT)//(BATCH_SIZE))

# Why do we need to make steps = 1 to predict our validation set ?????????? maybe to predict all the data we got!!!

pred = model.predict(X_test,verbose=1)

print(len(pred))
from sklearn.metrics import classification_report, confusion_matrix

# to_categorical(np.argmax(pred, axis=1))

def decode(datum):

    return np.argmax(datum)



mybatchlabels = []

for i in range(len(X_test)):

  mybatchlabels.append(decode(y_test[i]))



y_pred = np.argmax(pred, axis=1)

# print('Confusion Matrix')

# print(confusion_matrix(X_test, mybatchlabels))

print('Classification Report')

target_names = ['normal-pylorus', 'normal-z-line', 'dyed-resection-margins', 'ulcerative-colitis', 'dyed-lifted-polyps', 'normal-cecum', 'esophagitis', 'polyps']

print(classification_report(mybatchlabels, y_pred, target_names=target_names))
