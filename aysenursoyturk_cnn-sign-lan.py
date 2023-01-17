# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

import warnings 

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x_1 = np.load('../input/Sign-language-digits-dataset/X.npy')

y_1 = np.load('../input/Sign-language-digits-dataset/Y.npy')

X=np.concatenate((x_1[204:409], x_1[822:1027]),axis=0)

z = np.zeros(205)

o = np.ones(205)

Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1)

print("X shape: ",X.shape)

print("Y shape: ",Y.shape)
x1 = X.reshape(-1,64,64,1)

y1 = Y
print("x1 shape: ",x1.shape)

print("y1 shape: ",y1.shape)

from keras.utils.np_utils import to_categorical

y1 = to_categorical(y1,num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x1,y1,test_size=0.20,random_state=42)

print("x test: ",X_test.shape)

print("x train: ",X_train.shape)

print("y test: ",Y_test.shape)

print("y train: ",Y_train.shape)
from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator





model = Sequential()

model.add(Conv2D(filters=8,kernel_size=(5,5),padding = 'Same',activation='relu',input_shape=(64,64,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding='Same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))
optimizer = Adam(lr = 0.001,beta_1 =0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss = "categorical_crossentropy",metrics=["accuracy"])
epochs = 100

batch_size = 250
datagen = ImageDataGenerator()

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,Y_test), steps_per_epoch=X_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'],color = 'r',label = "test loss")

plt.title("test loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()