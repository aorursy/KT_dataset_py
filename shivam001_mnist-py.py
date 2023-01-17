# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.
# load dataset
train_dir=('../input/train.csv')
test_dir=('../input/test.csv')

train_data=pd.read_csv(train_dir)
test_data=pd.read_csv(test_dir)

x_train=np.array(train_data.iloc[:,1:])
y_train=to_categorical(np.array(train_data.iloc[:,0]))
x_test=np.array(test_data.iloc[:,0:])

print(x_train)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
# just for the shake of simplicty of the model all pixel value divide by 255 to make easy prediction on datasets
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32') /255
print(x_test.shape)
# import alll dependencies 
# you may write keras as  ->  from tensorflow import keras

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
# model name
model=Sequential()

# all all layers
# Convolution2D
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
# MaxPooling2D
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten
model.add(Flatten())
# deep layers
model.add(Dense(activation='relu',units=128))
#output layer
model.add(Dense(activation='softmax',units=10))
# compilation
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#training
print("Training............................................")
model.fit(x_train,y_train,epochs=10,batch_size=32)
print("\nTraining Completd !")
# predict class
prediction=model.predict_classes(x_test)
print("predict classes \n",prediction)
y_test=prediction
print("y_test \n",y_test)
