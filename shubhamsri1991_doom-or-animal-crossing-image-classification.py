# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.utils import plot_model, to_categorical

from tensorflow.keras import models, layers

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt

import os

import random

import cv2

from tensorflow.keras import callbacks

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = "../input/doom-crossing"

sub_dir = os.listdir(data_dir)

sub_dir
categories = ['animal_crossing', 'doom']



for category in categories:

    path = os.path.join(data_dir, category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array)

        plt.show()

        break

    break
print(img_array)
img_array.shape
IMG_SIZE = 80

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

plt.imshow(new_array)

plt.show()
training_data = []



def create_training_data():

    for category in categories:

        path = os.path.join(data_dir, category)

        category_count = categories.index(category)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, category_count])

            except Exception as e:

                pass

        

create_training_data()       
print(len(training_data))
random.shuffle(training_data)
x = []

y = []



for features, label in training_data:

    x.append(features)

    y.append(label)
fix, axes = plt.subplots(nrows=10, ncols=5, figsize=(12,18))

axes = axes.flatten()

for i,ax in zip(range(50), axes):

    ax.imshow(x[i])

    ax.set_title(categories[y[i]])

    ax.axis("off")

    

plt.show()
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = to_categorical(y)

x.shape, y.shape
model = models.Sequential()

model.add(layers.Flatten(input_shape=(80,80,1)))

model.add(layers.Dense(3200, activation='relu'))

model.add(layers.Dense(1600, activation='relu'))

model.add(layers.Dense(800, activation='relu'))

model.add(layers.Dense(400, activation='relu'))

model.add(layers.Dense(200, activation='relu'))

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(50, activation='relu'))

model.add(layers.Dense(2,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
NAME = "Animal-vs-Doom-Crossing"

checkpoint_path = "train_ckpt/cp.ckpt"

checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

callbacks_list = ([checkpoint], [tensorboard])

model.fit(x, y, epochs=50, batch_size=110, shuffle=True, validation_split=0.33, callbacks=callbacks_list, verbose=1)
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)