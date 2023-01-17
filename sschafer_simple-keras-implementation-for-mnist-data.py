

# Generic imports:

import numpy as np

import pandas as pd

import seaborn as sns

import os

import matplotlib.pyplot as plt

%matplotlib inline



# ML specific imports:

from sklearn.preprocessing import normalize

from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras.optimizers import Adadelta

from keras.losses import categorical_crossentropy



# Import data set

train = pd.read_csv('../input/train.csv')



labels = train.ix[:,0].values.astype('int32')

X_train = (train.ix[:,1:].values).astype('float32')

X_test = (pd.read_csv('../input/test.csv').values).astype('float32')


# create categorical matrix for train labels

y_train = np_utils.to_categorical(labels) 



# normalize data

X_train = normalize(X_train, norm='l2', axis=0)

X_test = normalize(X_test, norm='l2', axis=0)



#reshape X for covnet

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



# parameters:

batch_size = 512

n_classes = 10 # hopefully 10 digits

epochs = 20

in_dim = X_train.shape[1] # 28x28 pixels

input_shape = (28, 28, 1)







# Create a model with several layers, from Keras docs

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(n_classes, activation='softmax'))



model.compile(loss=categorical_crossentropy,

              optimizer=Adadelta(),

              metrics=['accuracy'])
if False: # comment out to avoid kernel timeout

    print("train...")

    hist = model.fit(X_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              verbose=1)

    print("get predictions...")

    preds = model.predict_classes(X_test, verbose=0)

    

    plt.figure()

    plt.plot(hist.history['acc'], '-o',label='Accuracy'); plt.legend()