# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading the files

X = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Spliting the data into features and labels

X_train = X.drop('label',axis=1)

y_train = X['label']



#Deleting to free up space

del X
#Making the values to be float so that it takes more values while scaling

X_train = X_train.astype('float32')

test = test.astype('float32')
#Scaling the data

X_train/=255

test/=255
#Reshaping the data to feed into keras model

X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

test = test.values.reshape(test.shape[0],28,28,1)
from keras.utils import to_categorical

y_train = to_categorical(y_train,num_classes=10)
#Importing the required libraries for building the neural net

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

from keras.callbacks import EarlyStopping
#Defining the model and its layers

model = Sequential()

model.add(Conv2D(28,kernel_size=3,input_shape=(28,28,1),activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(56,kernel_size=3,activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.summary() #summary of the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #compiling
early_stop = EarlyStopping(monitor='loss',min_delta=0.0001,patience=3) #Early stopping parameter
model.fit(X_train,y_train,epochs=40,batch_size=256,) #Fitting the model 
results = model.predict(test) #Predicting for the test set
predictions = results.argmax(axis=1)



predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)



submission.to_csv("simple_cnn.csv",index=False)