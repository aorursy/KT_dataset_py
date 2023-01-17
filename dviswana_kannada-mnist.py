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
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sat Jan 11 22:14:30 2020



@author: deepa

"""

import numpy as np

import  pandas as pd

from keras.models import Sequential

from keras.layers.core import Dense, Flatten, Dropout, Activation

#from keras.layers.convolutional import Conv2D, Maxpooling2D

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

np.random.seed(101)

NB_EPOCH=20

BATCH_SIZE=128

NB_CLASSES=10

VERBOSE=1

N_HIDDEN=128

DROPOUT=.3

VALIDATION_SPLIT=.3

OPTIMIZER=SGD()

X_train=pd.read_csv("../input/Kannada-MNIST/train.csv")

y_train=X_train.loc[:,'label']

X_train=X_train.iloc[:,1:]

print(X_train.shape)

print(y_train.shape)

print("********************ytarin********************************")

for i in range(0,10):

    print(y_train[i])



X_test=X_train.iloc[42001:60000,:]

y_test=y_train.iloc[42001:60000]

X_train=X_train.iloc[0:42000,:]

y_train=y_train.iloc[0:42000]

print(X_train.shape)

print(y_train.shape)

print("********************ytarin********************************")

for i in range(0,10):

    print(y_train[i])

print(X_test.shape)

print(y_test.shape)

print("********************ytest********************************")

print(y_test)  

  

X_test2=pd.read_csv("../input/Kannada-MNIST/test.csv")

y_id2=X_test2.loc[:,'id']

X_test2=X_test2.iloc[:,1:]

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

X_train=X_train.astype("float32")

X_test=X_test.astype("float32")

X_test2=X_test2.astype("float32")

X_train=X_train/255

X_test=X_test/255

X_test2=X_test2/255

y_train=np_utils.to_categorical(y_train, NB_CLASSES)

y_test=np_utils.to_categorical(y_test, NB_CLASSES)

print("********************ytarin********************************")

for i in range(0,10):

    print(y_train[i])

  

model=Sequential()

model.add(Dense(N_HIDDEN, input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dropout(DROPOUT))

model.add(Dense(N_HIDDEN, input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dropout(DROPOUT))

model.add(Dense(NB_CLASSES,input_shape=(784,)))

model.add(Activation('softmax'))

model.summary()

model.compile(optimizer=OPTIMIZER,loss="categorical_crossentropy",metrics=['accuracy'])

model.fit(X_train, y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score=model.evaluate(X_test, y_test,verbose=VERBOSE)

print("test score: ", score[0])

print("test accuracy:",score[1])

prediction=model.predict_classes(X_test2, verbose=0)

prediction=pd.DataFrame(prediction)

prediction['id']=y_id2

prediction.to_csv('/kaggle/working/predictions1.csv')