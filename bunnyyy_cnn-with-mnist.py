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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator
train= pd.read_csv("../input/digit-recognizer/train.csv")

test= pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
Y_train= train['label'] 
X_train= train.drop(labels= 'label', axis=1)
p = sns.countplot(Y_train)
X_train= X_train/255.0

test= test/255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train
Y_train= to_categorical(Y_train, num_classes=10)
Y_train
X_train, X_val, Y_train, Y_val= train_test_split(X_train, Y_train, test_size= 0.1, random_state= 2) 
from keras.callbacks import ReduceLROnPlateau

model= Sequential()

model.add(Conv2D(32, (5,5), padding= 'Same', activation= 'relu', input_shape= (28,28,1)))

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



model.compile(Adam(lr= 0.001), loss='categorical_crossentropy',

                  metrics=['accuracy'])

#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

  #                                          patience=3, 

   #                                         verbose=1, 

     #                                       factor=0.5, 

      #                                      min_lr=0.00001)



epochs= 10

batch_size= 64
from keras.preprocessing import image

gen = image.ImageDataGenerator()

batches = gen.flow(X_train, Y_train, batch_size=64)

val_batches=gen.flow(X_val, Y_val, batch_size=64)
history = model.fit_generator(generator= batches,

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])