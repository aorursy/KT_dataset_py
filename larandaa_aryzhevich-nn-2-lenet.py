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
root_path = '/kaggle/input/Kannada-MNIST/'

output_path = '/kaggle/working/'
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
df = pd.read_csv(root_path + 'train.csv')

df.shape
X_data = df.values[:, 1:]

y_data = df.label.values



X_data.shape, y_data.shape
df2 = pd.read_csv(root_path + 'Dig-MNIST.csv')

df2.label.value_counts()
test_X = df2.values[:, 1:]

test_y = df2.label.values



test_X.shape, test_y.shape
size = X_data.shape[0]

perm = np.random.permutation(size)



X_data = X_data[perm]

y_data = y_data[perm]



train_size = int(size / 10 * 7.5)

train_X, valid_X = X_data[:train_size], X_data[train_size:]

train_y, valid_y = y_data[:train_size], y_data[train_size:]



train_X.shape, train_y.shape, test_X.shape, test_y.shape
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)

valid_X = valid_X.reshape(valid_X.shape[0], 28, 28, 1)

test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)



train_X = np.pad(train_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

valid_X = np.pad(valid_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

test_X = np.pad(test_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')



mean_px = train_X.mean().astype(float)

std_px = train_X.std().astype(float)

train_X = (train_X - mean_px) / (std_px)

valid_X = (valid_X - mean_px) / (std_px)

test_X = (test_X - mean_px) / (std_px)



train_X.shape, valid_X.shape, test_X.shape
from keras.utils.np_utils import to_categorical



train_y = to_categorical(train_y)

valid_y = to_categorical(valid_y)
import keras

from keras.layers import *

from keras.models import Model

from keras.callbacks import CSVLogger, ModelCheckpoint
model = keras.Sequential()



model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'tanh', input_shape = (32,32,1)))

model.add(AveragePooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'tanh', input_shape = (14,14,6)))

model.add(AveragePooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(filters = 120, kernel_size = 5, strides = 1, activation = 'tanh', input_shape = (5, 5, 16)))

model.add(Flatten())

model.add(Dense(units = 84, activation = 'tanh'))

model.add(Dense(units = 10, activation = 'softmax'))



model.compile(optimizer ='sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(train_X, train_y,

          batch_size=16,

          epochs=30,

          verbose=1,

          validation_data=(valid_X, valid_y),

          callbacks=[

              CSVLogger(output_path + 'log.csv'),

              ModelCheckpoint(output_path + 'model.h5', save_best_only=True),

          ])
import keras

model = keras.models.load_model(output_path + 'model.h5')

pred_probas = model.predict(test_X, batch_size=16)

prediction = pred_probas.argmax(axis=1)
from sklearn.metrics import accuracy_score

accuracy_score(test_y, prediction)
df3 = pd.read_csv(root_path + 'test.csv')
ids = df3.id.values

res_X = df3.values[:, 1:]



res_X.shape, ids.shape
res_X = res_X.reshape(res_X.shape[0], 28, 28, 1)

res_X = np.pad(res_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

res_X = (res_X - mean_px) / (std_px)
pred_probas = model.predict(res_X, batch_size=16)

prediction = pred_probas.argmax(axis=1)



prediction[:10], prediction.shape
submission = pd.read_csv(root_path + 'sample_submission.csv')

submission['label'] = prediction



submission.head()
submission.to_csv("submission.csv", index=False)