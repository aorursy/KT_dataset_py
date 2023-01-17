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
#import the necessary libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam,RMSprop

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils,plot_model

from keras.preprocessing.image import ImageDataGenerator

from IPython.display import Image

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from keras.datasets import mnist
train_mnist = pd.read_csv('../input/digit-recognizer/train.csv')

test_mnist = pd.read_csv('../input/digit-recognizer/test.csv') 
X = train_mnist.drop("label",axis=1)

y = train_mnist['label']
X.shape
X = X / 255.0

test_mnist = test_mnist / 255.0
X = pd.DataFrame(X)

y = pd.DataFrame(y)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)
X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

X_test = X_test.values.reshape(X_test.shape[0],28,28,1)

test_mnist = test_mnist.values.reshape(test_mnist.shape[0],28,28,1)
model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=(28,28,1),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128,(3, 3),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(192,(3, 3),padding="SAME"))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Flatten())



# Fully connected layer

model.add(Dense(256))

# model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(10))



model.add(Activation('softmax'))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3, verbose=2,factor=0.5,min_lr=0.00001)

best_model = ModelCheckpoint('mnist_weights.h5', monitor='val_acc', verbose=2, save_best_only=True, mode='max')



early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10,restore_best_weights=True)
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
gen = ImageDataGenerator(

    featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        rotation_range=10, 

        zoom_range = 0.1,

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False,

        vertical_flip=False)



gen.fit(X_train)
h = model.fit_generator(

    gen.flow(X_train, y_train, batch_size=64),

    validation_data=(X_test, y_test),

    steps_per_epoch=len(X_train) // 64,

    epochs=50, verbose=1,

    callbacks=[learning_rate_reduction,best_model,early_stopping]

    )

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis = 1)

accuracy_score(y_test,y_pred)
y_pred[:20]
pd.DataFrame(h.history).plot()

conf_mat = confusion_matrix(y_test,y_pred)

f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(conf_mat, cmap='Blues',annot=True, linewidths=.5, fmt= '.1f',ax=ax)
pred = model.predict(test_mnist)

pred = np.argmax(pred,axis = 1)
HV_submission = pd.DataFrame({'ImageId': range(1,len(test_mnist)+1) ,'Label':pred })
HV_submission.to_csv("hv_submission1.csv",index=False)