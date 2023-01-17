# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import listdir

from os.path import isfile, join

import glob

import numpy as np

import cv2

import time

import matplotlib.pyplot as plt  # Just so we can visually confirm we have the same images

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def map_chars(path):



  map_characters = {}

  i = 0

  for fold in glob.glob(path + '*', recursive=False):

    map_characters[i] = fold.split('/')[-1]

    i += 1

  return(map_characters)

def load_train_set(path, map_characters=None, size=(64, 64), 

                   n_samples=None, verbose=1, print_step = 50):



  if (map_characters == None):

    map_characters = map_chars(path)



  pics, labels = [], []

  time_start = time.time()

  

  for (i, char) in map_characters.items():

    if(verbose > 0):

      print('Loading {}, time {}'.format(char, time.time() - time_start))

    for j, image_path in enumerate(glob.glob(path + char + '/*.*')):

      try:

        temp = cv2.imread(image_path)

        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        temp = cv2.resize(temp,(size[0],size[1])).astype('float32') / 255.

        if ((verbose == 1) and (j % print_step == 0)):

          print('\tStep {}:, time {}'.format(j, time.time() - time_start))

        if (j == n_samples):

          break

        pics.append(temp)

        labels.append(char)

      except:

        print('\tStep {}: Error'.format(j))

        continue

  print('Done. Total time: {}'.format(time.time() - time_start))

  print('Done. Total imgs: {}'.format(len(pics)))

  return(np.array(pics), labels)

X, y = load_train_set(path=('../input/trash-dataset/trash_dataset/'), verbose=1, print_step = 50)

np.save('X_data.npy', X)

np.save('y_data.npy', y)
X = np.load('X_data.npy')

y = np.load('y_data.npy')
from sklearn.preprocessing import OneHotEncoder

oneHE = OneHotEncoder(sparse=False)

y = oneHE.fit_transform(y.reshape(-1, 1))
y
plt.figure(figsize=(20,10))

for i in range(4):

    plt.subplot(1, 4, i + 1)

    n = np.random.choice(X.shape[0])

    

    plt.imshow(X[n])

    plt.title(oneHE.inverse_transform(y[n].reshape(1, -1))[0][0])
import keras

from keras.applications.vgg16 import VGG16
vgg = VGG16(weights='../input/vgg16-with-top-weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
vgg.summary()
model = keras.models.Sequential()



model.add(keras.layers.InputLayer(input_shape=(64, 64, 3)))

for layer in vgg.layers[1:-3]:

    model.add(layer)

model.add(keras.layers.Dense(512, activation='relu', name='hidden1'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(256, activation='relu', name='hidden2'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(4, activation='softmax', name='predictions'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer= keras.optimizers.SGD(lr=0.005, momentum=0.5),metrics=['accuracy'])
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

X_train, X_val, y_train ,y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
aug = ImageDataGenerator(rotation_range=15, 

                         zoom_range=0.20, 

                         width_shift_range=0.2, 

                         height_shift_range=0.2, 

                         shear_range=0.15,

                          horizontal_flip=True, 

                         fill_mode="nearest")

aug = aug.flow(X_train, y_train)
plt.imshow(aug[1][0][19])
try:

    history = model.fit(x=aug, epochs = 50, shuffle = True, validation_data=(X_val, y_val))

except KeyboardInterrupt:

    print('\n\nStopped')
import sklearn

from sklearn.metrics import classification_report



print('\n', sklearn.metrics.classification_report(np.argmax(y_val, axis=1), 

                                                  np.argmax(model.predict(X_val), axis=1), 

                                                  target_names=['plastic', 'glass', 'metal', 'paper']))
model.predict(X_test)
y_test_pred=oneHE.inverse_transform(model.predict(X_test))
y_test_real=oneHE.inverse_transform(y_test)

print('\n', sklearn.metrics.classification_report(y_test_real, 

                                                  y_test_pred, 

                                                  target_names=['plastic', 'glass', 'metal', 'paper']))
model.save('mod.hdf5')