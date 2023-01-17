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
import keras

from keras  import models

from keras  import layers

from keras import optimizers
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_features = df_train.iloc[:, 1:785]

df_labels = df_train.iloc[:, 0]

X_test = df_test.iloc[:, 0:784]

print(X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_cv, Y_train, Y_cv = train_test_split(df_features, df_labels,

                                               test_size = 0.2,

                                                random_state = 1212

                                               )

print(X_train.shape)

X_train = X_train.as_matrix().reshape(33600, 784)

X_cv = X_cv.as_matrix().reshape(8400, 784)

X_test = X_test.as_matrix().reshape(28000, 784)
# Feature Normalization

X_train = X_train.astype('float32'); X_cv = X_cv.astype('float32'); X_test = X_test.astype('float32')

X_train /= 255; X_cv /= 255; X_test /= 255



# Convert Labels to one hot encoded

from keras.utils import to_categorical

num_digits = 10

Y_train = to_categorical(Y_train, num_digits)

Y_cv = to_categorical(Y_cv, num_digits)
model1 = models.Sequential()

model1.add(layers.Dense(300, activation = 'relu', input_shape = (784,)))

model1.add(layers.Dense(100, activation = 'relu'))

model1.add(layers.Dense(100, activation = 'relu'))

model1.add(layers.Dense(100, activation = 'relu'))

model1.add(layers.Dense(200, activation = 'relu'))

model1.add(layers.Dense(10, activation = 'softmax'))



print(model1.summary())
rms_prop = keras.optimizers.rmsprop(lr = 0.01)

model1.compile(optimizer = 'rmsprop',

              loss = 'categorical_crossentropy',

               metrics = ['accuracy']

              )

history = model1.fit(X_train,

                    Y_train,

                     epochs = 20,

                     batch_size = 1000,

                     validation_data = (X_cv, Y_cv),

                     verbose = 2

                    )

test_pred = pd.DataFrame(model1.predict(X_test, batch_size = 200))

test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

test_pred.index.name = 'ImageID'

test_pred = test_pred.rename(columns = {0 : 'Labels'}).reset_index()

test_pred['ImageID'] = test_pred['ImageID'] + 1

test_pred.head()
test_pred.to_csv('mnist_submission.csv', index = False)