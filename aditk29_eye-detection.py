# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from IPython.display import clear_output

from time import sleep

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Train_Dir = '../input/training/training.csv'

Test_Dir = '../input/test/test.csv'

lookid_dir = '../input/IdLookupTable.csv'

train_data = pd.read_csv(Train_Dir)  

test_data = pd.read_csv(Test_Dir)

lookid_data = pd.read_csv(lookid_dir)

os.listdir('../input')
train_data.fillna(method = 'ffill',inplace = True)

#train_data.reset_index(drop = True,inplace = True)

train_data.isnull().any().value_counts()
train_data.keys()


x_datat=[]

for x in range(7049):

    x_datat.append(np.array([float(train_data["left_eye_center_x"][x]),float(train_data["left_eye_center_y"][x])]))

x_data=[]

for x in range (7049):

    temp=train_data["Image"][x].split()

    temp = np.asarray(temp , 

        dtype = np.float64, order ='C') 

    x_data.append(np.array(temp).reshape(96,96))

y_data=[]

for x in range (len(x_data)):

    y_data.append(np.array([float(train_data["left_eye_inner_corner_x"][x]),float(train_data["left_eye_inner_corner_y"][x]),float(train_data["left_eye_outer_corner_x"][x]),float(train_data["left_eye_outer_corner_y"][x])]))
x_data=np.array(x_data)

x_datat=np.array(x_datat)

print(len(x_data))

num=int(len(x_data)*.9)

print(num)

x_test=x_data[num:]

x_data=x_data[:num]

x_testt=x_datat[num:]

x_datat=x_datat[:num]

y_data=np.array(y_data)

Y_data=y_data[num:]

y_test=y_data[:num]#[0:30]

x_data=x_data.reshape(len(x_data),96,96,1)

x_test=x_test.reshape(len(x_test),96,96,1)

#x_data=x_data[0:30]

print(x_data.shape)

print(Y_data.shape)

x_train=x_data
print(x_datat)
import cv2

import numpy as np

x_data=np.array(x_data)/256

#cv2.imshow('image', x_data[0])
print(x_train.shape[1:][1:])
import keras.models as models

import keras.layers as layers

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, BatchNormalization

from keras.optimizers import SGD
import tensorflow as tf
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint("keras.model1", savebest_only=True, verbose=1)


model2 = models.Sequential()

model2.add(Dense(12, input_dim=2, activation='relu'))

model2.add(Dense(8, activation='relu'))

model2.add(Dense(4))



model1 = models.Sequential()

model1.add(layers.Conv2D(filters=80, kernel_size=(5,5), activation='relu', input_shape=x_data.shape[1:]))

model1.add(layers.MaxPool2D(pool_size=(2, 2)))

model1.add(layers.Dropout(rate=0.6))

model1.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))

model1.add(layers.MaxPool2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(filters=24, kernel_size=(3, 3), activation='relu'))

# model.add(layers.MaxPool2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))

# model.add(layers.MaxPool2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

# model.add(layers.MaxPool2D(pool_size=(2, 2)))

model1.add(layers.Dropout(rate=0.6))

model1.add(layers.Flatten())

model1.add(layers.Dense(256, activation='relu'))

model1.add(layers.Dropout(rate=0.6))

model1.add(layers.Dense(4, activation='relu'))
# model = Sequential()



# model.add(BatchNormalization(input_shape=(96, 96, 1)))

# model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(96, 96, 1), dim_ordering="tf"))

# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))



# model.add(Convolution2D(36, 5, 5))

# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))



# model.add(Convolution2D(48, 5, 5))

# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))



# model.add(Convolution2D(64, 3, 3))

# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))



# model.add(Convolution2D(64, 3, 3))

# model.add(Activation('relu'))



# model.add(GlobalAveragePooling2D());



# model.add(Dense(500, activation="relu"))

# model.add(Dense(90, activation="relu"))

# model.add(Dense(4))
import tensorflow as tf

model1.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model1.fit(x_data, y_test, epochs=20,validation_split=.2,callbacks=[modelcheckpoint])
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint("keras.model2", savebest_only=True, verbose=1)
import tensorflow as tf

model2.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model2.fit(x_datat, y_test, epochs=20,batch_size=10,callbacks=[modelcheckpoint])


predictions = model1.predict(x_test)

results=predictions-Y_data



sum=0

count=0

for r in results:

    for re in r:

        sum+=abs(re)

        count+=1

avg=sum/count

print(avg)

print(results)


predictions = model2.predict(x_testt)

results=predictions-Y_data



sum=0

count=0

for r in results:

    for re in r:

        sum+=abs(re)

        count+=1

avg=sum/count

print(avg)

print(results)
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint("keras.model3", savebest_only=True, verbose=1)
# stacked generalization with linear meta model on blobs dataset

from sklearn.datasets import make_blobs

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from keras.models import load_model

from keras.utils import to_categorical

from numpy import dstack





# create stacked model input dataset as outputs from the ensemble

def stacked_dataset(members,inputX):

    stackX = None

    for m in range(len(members)):

        model=members[m]

        # make prediction

        yhat = model.predict(inputX[m], verbose=0)

        # stack predictions into [rows, members, probabilities]

        if stackX is None:

            stackX = yhat

        else:

            stackX = dstack((stackX, yhat))

    # flatten predictions to [rows, members x probabilities]

    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

    return stackX



# fit a model based on the outputs from the ensemble members

def fit_stacked_model(members, inputX, inputy):

    # create dataset using ensemble

    stackedX = stacked_dataset(members, inputX)

    # fit standalone model

    model = models.Sequential()

    model.add(Dense(36, input_dim=8, activation='relu'))

    model.add(Dense(24, activation='sigmoid'))

    model.add(Dense(18, activation='relu'))

    model.add(Dense(12, activation='relu'))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(4))



    return model



members=[model1,model2]

model3=fit_stacked_model(members,[x_data,x_datat],y_test)

input3=stacked_dataset(members,[x_data,x_datat])

import tensorflow as tf

model3.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model3.fit(input3, y_test, epochs=30,batch_size=15,callbacks=[modelcheckpoint])

predictions = model3.predict(stacked_dataset(members,[x_test,x_testt]))

results=predictions-Y_data



sum=0

count=0

for r in results:

    for re in r:

        sum+=abs(re)

        count+=1

avg=sum/count

print(avg)

print(results)
model3.save("Integrated_eye_detection")