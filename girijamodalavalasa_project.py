# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export
def load_data():
  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join('/kaggle/input/cifar-10-batches-py/', 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join('/kaggle/input/cifar-10-batches-py/', 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  x_test = x_test.astype(x_train.dtype)
  y_test = y_test.astype(y_train.dtype)

  return (x_train, y_train), (x_test, y_test)
(X_train, y_train) , (X_test, y_test)= load_data()
X_train=X_train[:50000]
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
i = 30009
plt.imshow(X_train[i])
print(y_train[i])
W_grid = 4
L_grid = 4
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel()
n_training = len(X_train)
for i in np.arange(0, L_grid * W_grid):
 index = np.random.randint(0, n_training) # pick a random number
 axes[i].imshow(X_train[index])
 axes[i].set_title(y_train[index])
 axes[i].axis('off')

plt.subplots_adjust(hspace = 0.4)
n_training


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
number_cat = 10
y_train
import keras
y_train = keras.utils.to_categorical(y_train, number_cat)
y_train

y_test = keras.utils.to_categorical(y_test, number_cat)
y_test
X_train = X_train/255
X_test = X_test/255
X_train
X_train.shape
Input_shape = X_train.shape[1:]
Input_shape
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 1024, activation = 'relu'))
cnn_model.add(Dense(units = 1024, activation = 'relu'))
cnn_model.add(Dense(units = 10, activation = 'softmax'))
cnn_model.compile(loss = 'categorical_crossentropy', optimizer= keras.optimizers.RMSprop(lr = 0.001), metrics=['accuracy'])
print(cnn_model.summary())
history = cnn_model.fit(X_train, y_train, batch_size = 250, epochs = 45, shuffle = True)
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test)
predicted_classes
y_test
y_test = y_test.argmax(1)
y_test
L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()
for i in np.arange(0, L*W):
 axes[i].imshow(X_test[i])
 axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
 axes[i].axis('off')
plt.subplots_adjust(wspace = 1) 
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, predicted_classes)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)
cnn_model.save('project.h5')
(X_train, y_train), (X_test, y_test) = load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train.shape
n = 8
X_train_sample = X_train[:n]
X_train_sample.shape
from keras.preprocessing.image import ImageDataGenerator
# dataget_train = ImageDataGenerator(rotation_range = 90)
# dataget_train = ImageDataGenerator(vertical_flip=True)
# dataget_train = ImageDataGenerator(height_shift_range=0.5)
dataget_train = ImageDataGenerator(brightness_range=(1,3))
dataget_train.fit(X_train_sample)