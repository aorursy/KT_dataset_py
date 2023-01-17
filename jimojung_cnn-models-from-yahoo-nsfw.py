from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras.utils import Sequence

from keras import backend as K

from keras.applications.resnet import ResNet50

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import tensorflow as tf

import os

import glob
all_files = glob.glob("/kaggle/input/yahoo*/*/*/*/*/*.npz")

#print(all_files)

print("file count=", len(all_files))
class NumPyFileGenerator(Sequence):

    def __init__(self, files):

        self.files = files



    def __len__(self):

        return len(self.files)



    def __getitem__(self, idx):

        data = np.load(open(self.files[idx], 'rb'), allow_pickle=True)

        #print("DATA= ", np.array(data))

        x = data['MobileNetV2_bottleneck_features']

        y = data['azure_output']

        y2 = y[:, [2]]

        #print("X dim= ", np.shape(x))

        #print("X= ", x)

        #print("y dim= ", np.shape(y))

        #print("y= ", y)

        #print("y2 dim= ", np.shape(y2))

        #print("y2= ", y2)

        return x, y2
training_generator = NumPyFileGenerator(files=all_files[0:3000])

validation_generator = NumPyFileGenerator(files=all_files[3000:3300])
def threshold_accuracy(y_true, y_pred):

    absolute_difference = K.abs(y_true - y_pred)

    truth_matrix = K.greater(absolute_difference, K.variable(0.05))

    casted = K.cast(truth_matrix, 'float32')

    final = K.mean(casted)

    return final
model = tf.keras.Sequential([

  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(1)

])



model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),

              loss='mean_squared_error', metrics=[threshold_accuracy])
print("training_generator=", len(training_generator))

print("validation_generator=", len(validation_generator))

epochs=30

history = model.fit_generator(

                    training_generator,

                    validation_data=validation_generator,

                    epochs=epochs,

                    steps_per_epoch=len(training_generator)/epochs,

                    validation_steps=len(validation_generator)/epochs,

                    verbose=2)
plt.plot(history.history['threshold_accuracy'])

plt.plot(history.history['val_threshold_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['Train', 'Test'], loc='upper left')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.yscale('log')

plt.show()