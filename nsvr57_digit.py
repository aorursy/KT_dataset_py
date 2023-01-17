# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kauggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential,load_model

from keras.layers import Dense, Activation

from keras.layers import *

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint

from keras import optimizers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.optimizers import RMSprop

import keras
digit=pd.read_csv('../input/train.csv')
digit.shape
digit.head()
Feature=digit.drop(columns='label')

Target=digit['label']
Xr,Xt,Yr,Yt=train_test_split(Feature,Target,test_size=0.10,random_state=1)
Xr=Xr/255

Xt=Xt/255
Xr.shape, Yr.shape
Xr=Xr.values.reshape(-1,28,28,1)

Xt=Xt.values.reshape(-1,28,28,1)

Yr=tf.keras.utils.to_categorical(Yr,num_classes=10)

Yt=tf.keras.utils.to_categorical(Yt,num_classes=10)
plt.imshow(Xr[1][:,:,0])
model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



# model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

# model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

# model.add(MaxPooling2D((2, 2)))

# model.add(BatchNormalization(axis=3))

# model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])



model.summary()







# early_stops=keras.callbacks.EarlyStopping(patience=5)



# call_backs=[early_stops]



# model.fit(Xr,Yr,batch_size=32,nb_epoch=10,validation_data=(Xt, Yt), callbacks=call_backs)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False)  # randomly flip images

datagen.fit(Xr)
early_stops=keras.callbacks.EarlyStopping(patience=50)

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')



model.fit_generator(datagen.flow(Xr,Yr,batch_size=86),

                     samples_per_epoch=Xr.shape[0],

                     nb_epoch=30

                     ,

                     validation_data=(Xt, Yt),callbacks=[early_stops,mcp_save])
test=pd.read_csv('../input/test.csv')
test=test.values.reshape(-1,28,28,1)
model=load_model('.mdl_wts.hdf5')
pred=model.predict(test)
result=np.argmax(pred,axis=1)
np.unique(result)
results = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit.csv",index=False)