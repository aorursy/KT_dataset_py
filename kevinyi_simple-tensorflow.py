# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

from tqdm import tqdm

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import cv2

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def load_data():

    img_dir = "../input/cell_images/cell_images"

    y_map = {"Uninfected":0, "Parasitized": 1}

    

    x_data = []

    y_data = []

    

    for dir_n in y_map:

        label = int(y_map[dir_n])

        for ip in tqdm(os.listdir(os.path.join(img_dir, dir_n))):

            try:

                x_data.append(cv2.resize(cv2.imread(os.path.join(img_dir, dir_n, ip)), (64, 64)))

                y_data.append(label)

            except:

                continue

                

    return np.array(x_data), np.array(y_data)
x_data, y_data = load_data()

y_data_cat = tf.keras.utils.to_categorical(y_data, num_classes=2)
def keras_model():    

    l = tf.keras.layers

    

    input_layer = l.Input(shape=(64, 64, 3))

    

    x = l.Conv2D(64, (3, 3), 2, activation="relu")(input_layer)

    x = l.BatchNormalization()(x)

    

    x = l.Conv2D(128, (3, 3), 2, activation="relu")(x)

    x = l.Dropout(0.4)(x)

    x = l.Conv2D(256, (3, 3), activation="relu")(x)

    x = l.MaxPooling2D()(x)

    

    x = l.Conv2D(1024, (3, 3), activation="relu")(x)

    x = l.Dropout(0.3)(x)

    

    x = l.Conv2D(512, (3, 3), activation="relu")(x)

    x = l.Dropout(0.4)(x)

    

    x = l.Flatten()(x)

    

    x = l.Dense(256, activation="relu")(x)

    x = l.Dropout(0.3)(x)

    

    x = l.Dense(2, activation="softmax")(x)

    

    model = tf.keras.models.Model(input_layer, x)

    

    model.summary()

    

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001, decay=0.01), metrics=["acc"])

    

    return model
model = keras_model()
history = model.fit(x_data, y_data_cat, batch_size=512, epochs=10, validation_split=0.1)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()