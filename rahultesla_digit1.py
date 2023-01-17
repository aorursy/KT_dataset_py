# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
y = data['label']
x = data.drop(['label'],axis=1)
testx = testdata
x = x/255.0
testx = testx/255.0 
y = to_categorical(y,num_classes=10)
x = x.values.reshape(-1,28,28,1)
testx = testx.values.reshape(-1,28,28,1)
del data
xtr,xval,ytr,yval=train_test_split(x,y,test_size = 0.1)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 30
batch_size = 86
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
datagen.fit(xtr)
h = model.fit_generator(datagen.flow(xtr,ytr, batch_size=batch_size),
                              epochs = epochs, validation_data = (xval,yval),
                              verbose = 1, steps_per_epoch=xtr.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction],)
testy = model.predict_classes(testx)
sub = pd.DataFrame({'ImageId' : list(range(1,len(testy)+1)),'Label' : testy})
sub.to_csv('submission.csv',index=False)
