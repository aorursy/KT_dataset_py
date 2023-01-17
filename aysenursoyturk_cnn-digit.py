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
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head(10)
test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head(10)
y_train = train["label"]

x_train = train.drop(labels = ["label"],axis=1)
x_train = x_train/255.0

test = test/255.0
print("x_train shape",x_train.shape)

print("test shape",test.shape)
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape",x_train.shape)

print("test shape",test.shape)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train,num_classes = 10)

print(y_train)
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.15,random_state=2)

print("x_train shape",x_train.shape)

print("x_test shape",x_val.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_val.shape)
plt.imshow(x_train[4][:,:,0],cmap = 'gray')

plt.show()

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()

model.add(Conv2D(filters=8,kernel_size=(5,5),padding = 'Same',activation='relu',input_shape=(28,28,1)))

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
epochs = 20

batch_size = 250
datagen = ImageDataGenerator()

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'],color = 'r',label = "validation loss")

plt.title("Validation loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()