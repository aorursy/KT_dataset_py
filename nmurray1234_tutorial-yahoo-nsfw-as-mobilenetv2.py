from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras.utils import Sequence

from keras import backend as K

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os

import glob
all_files = glob.glob("/kaggle/input/*/*")
class NumPyFileGenerator(Sequence):

    def __init__(self, files):

        self.files = files



    def __len__(self):

        return len(self.files)



    def __getitem__(self, idx):

        data = np.load(open(self.files[idx], 'rb'))

        x = data['MobileNetV2_bottleneck_features']

        y = data['yahoo_nsfw_output']

        return x, y
training_generator = NumPyFileGenerator(files=all_files[0:8])

validation_generator = NumPyFileGenerator(files=all_files[9:10])
model = tf.keras.Sequential([

  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(1)

])
def threshold_accuracy(y_true, y_pred):

    absolute_difference = K.abs(y_true - y_pred)

    truth_matrix = K.greater(absolute_difference, K.variable(0.05))

    casted = K.cast(truth_matrix, 'float32')

    final = K.mean(casted)

    return final
model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss='mean_squared_error', 

              metrics=[threshold_accuracy])
history = model.fit_generator(training_generator, 

                    epochs=20, 

                    validation_data=validation_generator)
plt.plot(history.history['threshold_accuracy'])

plt.plot(history.history['val_threshold_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['Train', 'Test'], loc='upper left')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')


