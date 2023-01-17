from keras.layers import *

from keras.models import Sequential
# Build a Model 

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

# model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()
import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(df.shape)

data = df.values

print(data.shape)

print(type(data))

XTrain = data[:,1:]

YTrain = data[:,0]
#Dataset

from keras.datasets import mnist

from keras.utils import to_categorical
def preprocess_data(X,Y):

    X = X.reshape((-1,28,28,1))

    X = X/255.0

    Y = to_categorical(Y)

    return X,Y



XTrain,YTrain = preprocess_data(XTrain,YTrain)

print(XTrain.shape,YTrain.shape)
# Model Compilation

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(XTrain,YTrain,epochs=49,validation_split=0.1,batch_size=210)
Xtest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

data = Xtest.values

Xtest = data.reshape((-1,28,28,1))

Xtest = Xtest/255.0

print(Xtest.shape)
res = model.predict(Xtest)

results = np.argmax(res,axis = 1)

print(results)
Label = pd.Series(results,name = 'Label')

ImageId = pd.Series(range(1,28001),name = 'ImageId')

submission = pd.concat([ImageId,Label],axis = 1)

submission.to_csv('/kaggle/working/output.csv',index = False)