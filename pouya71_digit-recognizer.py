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
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 

import pandas as pd 

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from keras.callbacks import callbacks

from keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_train="../input/digit-recognizer/train.csv"

data_test= "../input/digit-recognizer/test.csv"

df_train=pd.read_csv(data_train)

df_test=pd.read_csv(data_test)
df_train.head()
print(df_train.shape)

print(df_test.shape)
train_image=df_train.drop(['label'], axis = 1).to_numpy().reshape(42000, 28, 28,1).astype('float32')

train_image/=255.

test_image=df_test.to_numpy().reshape(28000, 28, 28,1).astype('float32')

test_image/=255.
print(train_image.shape)

print(test_image.shape)
from keras.utils import to_categorical

train_l=to_categorical(df_train['label'],10)

network=models.Sequential()

network.add(layers.Dense(512,activation='relu',input_shape=(28,28,1)))

network.add(layers.Dense(10,activation='softmax'))

network.add(layers.Flatten())

network.add(layers.Dense(512, activation = 'relu'))

network.add(layers.Dropout(0.3))

network.add(layers.Dense(10, activation = 'softmax'))

network.summary()
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

  """Stop training when the loss is at its min, i.e. the loss stops decreasing.



  Arguments:

      patience: Number of epochs to wait after min has been hit. After this

      number of no improvement, training stops.

  """



  def __init__(self, patience=0):

    super(EarlyStoppingAtMinLoss, self).__init__()



    self.patience = patience



    # best_weights to store the weights at which the minimum loss occurs.

    self.best_weights = None



  def on_train_begin(self, logs=None):

    # The number of epoch it has waited when loss is no longer minimum.

    self.wait = 0

    # The epoch the training stops at.

    self.stopped_epoch = 0

    # Initialize the best as infinity.

    self.best = np.Inf



  def on_epoch_end(self, epoch, logs=None):

    current = logs.get('loss')

    if np.less(current, self.best):

      self.best = current

      self.wait = 0

      # Record the best weights if current results is better (less).

      self.best_weights = self.model.get_weights()

    else:

      self.wait += 1

      if self.wait >= self.patience:

        self.stopped_epoch = epoch

        self.model.stop_training = True

        print('Restoring model weights from the end of the best epoch.')

        self.model.set_weights(self.best_weights)



  def on_train_end(self, logs=None):

    if self.stopped_epoch > 0:

      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
network.fit(train_image,train_l,epochs=15,batch_size=128)

#callbacks = [EarlyStoppingAtMinLoss()]