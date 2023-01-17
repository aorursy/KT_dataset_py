from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.datasets import mnist
import pandas as pd

import numpy as np
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train.head(10)
import seaborn as sns

f = sns.countplot(Y_train)

Y_train.value_counts()
import matplotlib.pyplot as plt

indec = 45

print(Y_train[indec])
X_train.shape, Y_train.shape
test = pd.read_csv('../input/digit-recognizer/test.csv')

test.head()
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
model=Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

LRR = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.00001)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
hist = model.fit(X_train,Y_train,epochs=20,batch_size = 86)
model.evaluate(X_train,Y_train)