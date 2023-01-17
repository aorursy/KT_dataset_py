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
file = "../input/ionosphere_data_kaggle.csv"

df = pd.read_csv(file)

dataset = df.values

dataset.shape

X = dataset[:,0:34]

Y = dataset[:,34]
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(Y)

y = encoder.transform(Y)
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(34, input_dim=34, init= 'normal' , activation= 'relu' ))

model.add(Dense(1, init= 'normal' , activation= 'sigmoid' ))

# Compile model

epochs = 50

learning_rate = 0.1

decay_rate = learning_rate / epochs

momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(loss= 'binary_crossentropy' , optimizer='sgd', metrics=[ 'accuracy' ])

# Fit the model

model.fit(X, y, validation_split=0.33, nb_epoch=epochs, batch_size=28, verbose=2)
import pandas

import pandas

import numpy

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder

from keras.callbacks import LearningRateScheduler

# learning rate schedule

def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 10.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate



# encode class values as integers

encoder = LabelEncoder()

encoder.fit(Y)

Y = encoder.transform(Y)

# create model

model = Sequential()

model.add(Dense(34, input_dim=34, init= 'normal' , activation= 'relu' ))

model.add(Dense(1, init= 'normal' , activation= 'sigmoid' ))

# Compile model

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss= 'binary_crossentropy' , optimizer='sgd', metrics=[ 'accuracy' ])

# learning schedule callback

lrate = LearningRateScheduler(step_decay)

callbacks_list = [lrate]

# Fit the model

model.fit(X, y, validation_split=0.33, nb_epoch=50, batch_size=28,callbacks=callbacks_list, verbose=2)