# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, LeakyReLU, Activation

from tensorflow.keras.optimizers import Adam

import seaborn as sns

import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

df_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df_train.head(5)
df_test.head()
df_submission.head()
plt.imshow(df_train.iloc[3].values[1:].reshape(28, 28), cmap = 'gray')
X_train = df_train.iloc[:,1:].values.reshape(-1, 28, 28, 1)

y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

X_test = df_test.values.reshape(-1, 28, 28, 1)
print('Number of samples: {} - after reshape: {}'.format(len(df_train), X_train.shape[0]))
sns.countplot(x = y_train.ravel())
print(X_train[0].min(), X_train[0].max())
X_train = X_train / 255.
X_test = X_test / 255.
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2)
model = Sequential([

    Input(shape = (28, 28, 1)),

    Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'same'),

    BatchNormalization(),

    Activation('relu'),

    Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'),

    BatchNormalization(),

    Activation('relu'),

    Conv2D(filters = 64, kernel_size = (3, 3),strides = (2, 2), padding = 'same'),

    BatchNormalization(),

    Activation('relu'),

    Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same'),

    BatchNormalization(),

    Activation('relu'),

    Flatten(),

    Dense(units = 10),

    Activation('softmax')

])

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 64, epochs = 15, verbose = 1)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 32, epochs = 7, verbose = 1)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])

model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 16, epochs = 5, verbose = 1)
out_test = model.predict_classes(X_test)
df_submiss = np.hstack((np.arange(1, out_test.shape[0] + 1,1).reshape(-1, 1), out_test.reshape(-1, 1)))
data = pd.DataFrame(data = df_submiss, columns = ['ImageId', 'Label'])
data.head()
data.to_csv('submission.csv', index = False)