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
import tensorflow as tf

import matplotlib.pyplot as plt
#Defined my class

class myCallback(tf.keras.callbacks.Callback):

    def __init__(self, acc_threshold = 0.9, print_msg = True):

        self.acc_threshold = acc_threshold

        self.print_msg = print_msg

    

    def on_epoch_end(self, epoch, logs= {}):

        if (logs.get("acc") > self.acc_threshold):

            if self.print_msg:

                print("\nReached 90% accuracy so cancelling the training!")

            self.model.stop_training = True

        else:

            if self.print_msg:

                print("\nAccuracy not high enough. Starting another epoch...\n")

                
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
callbacks = myCallback()
def build_model(num_layers = 1, architecture = [32], act_func = "relu", 

               input_shape = (28, 28), output_class = 10):

    layers = [tf.keras.layers.Flatten(input_shape= input_shape)]

    if act_func == 'relu':

        activation = tf.nn.relu

    elif act_func == "sigmoid":

        activation = tf.nn.sigmoid

    elif act_func == "tanh":

        activation = tf.nn.tanh

    

    for i in range(num_layers):

        layers.append(tf.keras.layers.Dense(architecture[i], activation = tf.nn.relu))

    

    layers.append(tf.keras.layers.Dense(output_class, activation = tf.nn.softmax))

    

    model = tf.keras.models.Sequential(layers)

    return model
model = build_model(num_layers= 1, architecture=[128])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])