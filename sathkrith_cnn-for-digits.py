# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
from tensorflow import keras

from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf



num_classes = 10

img_rows = 28

img_cols = 28



def raw_to_img_array(raw_data):

    y = np.array(raw_data.iloc[:, 0])

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = np.array(raw_data.iloc[:,1:])

    num_images = raw_data.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y

from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.2, 

        shear_range=20, 

        width_shift_range=0.2,  

        height_shift_range=0.2,  

        horizontal_flip=False, 

        vertical_flip=False,

        )
from sklearn.model_selection import train_test_split



train_x, train_y = raw_to_img_array(train_data)

train_x,val_x,train_y,val_y=train_test_split(train_x, train_y, test_size=0.2)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

kannada_model = Sequential()

kannada_model.add(Conv2D(128, kernel_size=(4, 4),

                 activation='relu',padding="same",

                 input_shape=(img_rows, img_cols, 1)))

kannada_model.add(BatchNormalization())

kannada_model.add(MaxPool2D(pool_size=2,padding="valid"))

kannada_model.add(Dropout(0.25))

kannada_model.add(Conv2D(256,kernel_size=(4, 4),padding="same", activation='relu'))

kannada_model.add(BatchNormalization())

kannada_model.add(MaxPool2D(pool_size=2,padding="valid"))

kannada_model.add(Dropout(0.25))

kannada_model.add(Conv2D(512,kernel_size=(4, 4),padding="same", activation='relu'))

kannada_model.add(BatchNormalization())

kannada_model.add(MaxPool2D(pool_size=2,padding="valid"))

kannada_model.add(Dropout(0.25))

kannada_model.add(Flatten())

kannada_model.add(Dense(1024, activation='relu'))

kannada_model.add(BatchNormalization())

kannada_model.add(Dropout(0.4))

kannada_model.add(Dense(256, activation='relu'))

kannada_model.add(BatchNormalization())

kannada_model.add(Dropout(0.4))

kannada_model.add(Dense(num_classes, activation='softmax'))



kannada_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer="adam",

              metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

checkpoint = ModelCheckpoint("bestmodel.model", monitor='val_acc', verbose=1, save_best_only=True)

batch_size=64

kannada_model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),

          steps_per_epoch=train_x.shape[0]//batch_size,

          epochs=50,callbacks= [checkpoint,learning_rate_reduction],

          validation_data=datagen.flow(val_x,val_y,batch_size=batch_size),

                           validation_steps = val_x.shape[0]//batch_size)
test_x = np.array(test_data.iloc[:,1:])

test_index = np.array(test_data.iloc[:,0])

num_images = test_data.shape[0]

test_x = test_x.reshape(num_images, img_rows, img_cols, 1)

test_x = test_x / 255

kannada_model.load_weights('bestmodel.model')

preds = kannada_model.predict(test_x)
submissions = np.zeros((len(preds),2),dtype=int)

for ix in range(preds.shape[0]):

    submissions[ix][0] = test_index[ix]

    submissions[ix][1] = np.argmax(preds[ix])

print(submissions)
submissions_pd = pd.DataFrame(data=submissions,columns=["id","label"])

submissions_pd.head()

submissions_pd.to_csv("submissions.csv",index=False)