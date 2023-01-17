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
#%tensorflow_version 1.x

import tensorflow

print(tensorflow.__version__)

import tensorflow.keras as keras

print(keras.__version__)

import pandas



csvfile = pandas.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

array = csvfile.values

sourcelabels, source = array[:, :1], array[:, 1:]



csvfile = pandas.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

array = csvfile.values

testlabels, test = array[:, :1], array[:, 1:]
import sklearn.preprocessing



scaler = sklearn.preprocessing.MinMaxScaler((-1, 1), False)



scaler.fit(source)

ssource = scaler.transform(source)



scaler = sklearn.preprocessing.MinMaxScaler((-1, 1), False)



scaler.fit(test)

test = scaler.transform(test)
train, valid, trainlabels, validlabels = ssource[:50000], ssource[50000:],sourcelabels[:50000], sourcelabels[50000:]
from keras.layers import *

from keras.models import Model

from keras.callbacks import CSVLogger, ModelCheckpoint



x = Input(shape=(784,))

y = Dense(20, activation=None)(x)

y = Activation('elu')(y)

y = Dropout(rate=0.3)(y)

y = Dense(20, activation=None)(y)

y = Activation('elu')(y)

prediction = Dense(10, activation='softmax')(y)



model = Model(inputs=[x], output=[prediction])



model.compile(optimizer='sgd',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
model.fit(train, trainlabels,

          batch_size=16,

          epochs=20,

          verbose=1,

          validation_data=(valid, validlabels),

          callbacks=[CSVLogger('logs.csv'), ModelCheckpoint('model', save_best_only=True)])
model = keras.models.load_model('model')



pred = model.predict(test, batch_size=16)



prediction = pred.argmax(axis=1)
f = open("submission.csv", "w")



import csv

with f:

  writer = csv.writer(f)

    

  fnames = ["id", "label"]

  writer = csv.DictWriter(f, fieldnames=fnames) 

  writer.writeheader()



  for pos, value in enumerate(prediction):

    writer.writerow({"id" : pos, "label": value})