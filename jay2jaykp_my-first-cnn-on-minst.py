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
# import tensorflow as tf

# from tensorflow import keras

from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical

from keras.layers import Convolution2D, MaxPooling2D

from keras.models import  Sequential

from keras.layers.core import  Dense, Flatten, Dropout





from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#seperate out label and features

Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1) 



#reshape the data

X_train = X_train.values.reshape(X_train.shape[0], 28, 28,1)

test = test.values.reshape(test.shape[0], 28, 28,1)



# normalization of data

X_train = X_train / X_train.max()

test = test / test.max()



# one-hot encoding

Y_train= to_categorical(Y_train)

X_train.shape
model = Sequential()

model.add(Convolution2D(32,(5,5), activation='relu'))

model.add(Convolution2D(32,(5,5), activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.20))

model.add(Convolution2D(64, (3,3), activation='relu'))

model.add(Convolution2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))





#compile

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])



#fit

CNN = model.fit(X_train, Y_train, epochs=3, batch_size=16, validation_split=0.1)
#model prediction

predictions = model.predict_classes(test)
#submission



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)