# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape, test.shape)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train.iloc[:, 1:].values,

                                                  train.iloc[:, 0].values, 

                                                 test_size = 0.2, 

                                                 random_state = 42,

                                                 stratify = train.iloc[:, 0])

print("Training Data Shape = ", x_train.shape, y_train.shape)

print("Validation Data Shape = ", x_val.shape, y_val.shape)
fig, ax = plt.subplots(2, 1, sharex = 'col', sharey = 'row')

fig.set_size_inches(9, 10)   #width, height

sns.countplot(y_train, ax = ax[0])

ax[0].set_title("Training Data Labels")

sns.countplot(y_val, ax = ax[1])

ax[1].set_title("Validation Data Labels")
from keras import utils

y_train = utils.to_categorical(y_train, 10).astype(np.uint8)

y_val = utils.to_categorical(y_val, 10).astype(np.uint8)

print(y_train.shape)

print(y_train[0:5, :])
x_train = x_train/255.0

x_val = x_val/255.0

x_test = test.values/255.0
from keras import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import *

from keras.metrics import *
img_size = 784 #28*28

img_wid = 28

inputs = Input(shape = (img_size,))

F1 = Dense(units = 32, activation = 'relu')(inputs)

outputs = Dense(units=10, activation= 'softmax')(F1)

model = Model(inputs = [inputs], outputs = [outputs])

model.compile(optimizer = 'sgd', 

              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
epochs = 10

batch_size = 32



model.fit(x_train, y_train, batch_size= batch_size, epochs = epochs)
y_pred = model.predict(x_test).argmax(axis = 1)

print(y_pred.shape, np.unique(y_pred))
valScore, valAcc = model.evaluate(x_val, y_val)

trainScore, trainAcc = model.evaluate(x_train, y_train)

print("Training Score = ", trainScore, "Training Accuracy = ", trainAcc)

print("Validation Score = ", valScore, "Validation Accuracy = ", valAcc)
sub = pd.DataFrame()

sub['ImageId'] = np.arange(1, y_pred.shape[0] + 1)

sub['Label'] = y_pred

sub.to_csv("nnSimple.csv",index = False, header = True)
img_wid = 28



x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

inputs = Input(shape = (img_wid, img_wid, 1))



C1 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(inputs)

D1 = Flatten()(C1)

F1 = Dense(units = 32, activation = 'relu')(D1)

outputs = Dense(units = 10, activation = 'softmax')(F1)

model = Model(inputs = [inputs], outputs = [outputs])

model.compile(optimizer = 'sgd', 

              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
epochs = 10

batch_size = 32



model.fit(x_train, y_train, batch_size= batch_size, epochs = epochs)
y_pred = model.predict(x_test.reshape(x_test.shape[0], 28, 28, 1)).argmax(axis = 1)

print(y_pred.shape, np.unique(y_pred))
valScore, valAcc = model.evaluate(x_val, y_val)

trainScore, trainAcc = model.evaluate(x_train, y_train)

print("Training Score = ", trainScore, "Training Accuracy = ", trainAcc)

print("Validation Score = ", valScore, "Validation Accuracy = ", valAcc)
sub = pd.DataFrame()

sub['ImageId'] = np.arange(1, y_pred.shape[0] + 1)

sub['Label'] = y_pred

sub.to_csv("cnnSimple.csv",index = False, header = True)
img_wid = 28



x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

inputs = Input(shape = (img_wid, img_wid, 1))



#IMG = (28, 28), 

C1 = Conv2D(16, (3, 3), activation = 'relu', padding = "same")(inputs)

P1 = MaxPooling2D(pool_size=(2, 2))(C1)



#IMG = (14, 14)

C2 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(P1)

P2 = Dropout(0.15)(P1)

P2 = MaxPooling2D(pool_size= (2, 2))(C2)



#IMG = (7, 7)

C3 = Conv2D(128, (5, 5), activation = 'relu', padding = "same")(P2)

C3 = Dropout(0.15)(C3)

F1 = Flatten()(C3)



#Fully Connected Layer

D1 = Dense(units = 512, activation = 'relu',

           activity_regularizer=regularizers.l2(0.01))(F1)

D1 = Dropout(0.25)(D1)

D2 = Dense(units = 512, activation = 'relu',

           activity_regularizer=regularizers.l2(0.01))(D1)

D2 = Dropout(0.25)(D2)

outputs = Dense(units = 10, activation = 'softmax')(D2)



model = Model(inputs = [inputs], outputs = [outputs])

model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',

             metrics = ['accuracy'], )

model.summary()
import datetime

print(datetime.datetime.now())

epochs = 30

batch_size = 32



model.fit(x_train, y_train, batch_size= batch_size, epochs = epochs,

          validation_data = (x_val, y_val))

print(datetime.datetime.now())
y_pred = model.predict(x_test.reshape(x_test.shape[0], 28, 28, 1)).argmax(axis = 1)

print(y_pred.shape, np.unique(y_pred))
valScore, valAcc = model.evaluate(x_val, y_val)

trainScore, trainAcc = model.evaluate(x_train, y_train)

print("Training Score = ", trainScore, "Training Accuracy = ", trainAcc)

print("Validation Score = ", valScore, "Validation Accuracy = ", valAcc)
sub = pd.DataFrame()

sub['ImageId'] = np.arange(1, y_pred.shape[0] + 1)

sub['Label'] = y_pred

sub.to_csv("cnnLeNet-5.csv",index = False, header = True)