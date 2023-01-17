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
df.label.value_counts()
examples = df.values[:10, 1:]

plt.figure(1, figsize=(15, 6))

for i in range(10):

  plt.subplot(2, 5, i + 1)

  plt.imshow(examples[i].reshape((28, 28)), cmap='gray')
X_data = df.values[:, 1:]

y_data = df.label.values



X_data.shape, y_data.shape
df2 = pd.read_csv(root_path + 'Dig-MNIST.csv')

df2.label.value_counts()
examples = df2.values[:10, 1:]

plt.figure(1, figsize=(15, 6))

for i in range(10):

  plt.subplot(2, 5, i + 1)

  plt.imshow(examples[i].reshape((28, 28)), cmap='gray')
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
train_mean = np.mean(train_X)

train_X = train_X - train_mean



train_max = np.max(np.abs(train_X))

train_X = train_X / train_max



train_mean, train_max
valid_X = (valid_X - train_mean) / train_max

test_X = (test_X - train_mean) / train_max
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



model.compile(optimizer ='sgd',

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
model.fit(train_X, train_y,

          batch_size=16,

          epochs=20,

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
examples = df3.values[:10, 1:]

plt.figure(1, figsize=(15, 6))

for i in range(10):

  plt.subplot(2, 5, i + 1)

  plt.imshow(examples[i].reshape((28, 28)), cmap='gray')
df3.shape

df3.columns
ids = df3.id.values

res_X = df3.values[:, 1:]



res_X.shape, ids.shape
res_X = (res_X - train_mean) / train_max
pred_probas = model.predict(res_X, batch_size=16)

prediction = pred_probas.argmax(axis=1)



prediction[:10], prediction.shape
submission = pd.read_csv(root_path + 'sample_submission.csv')

submission['label'] = prediction



submission.head()
submission.to_csv("submission.csv", index=False)