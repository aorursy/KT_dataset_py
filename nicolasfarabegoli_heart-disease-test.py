# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset_path = pd.read_csv("../input/heart.csv")

dataset = dataset_path.copy()
dataset.head()
dataset.target.value_counts()
no_disease = len(dataset[dataset.target == 0])

have_disease = len(dataset[dataset.target == 1])



print("Percentage Patient have disease: {:.2f}%".format((have_disease / len(dataset.target) * 100)))

print("Percentage Patient haven't disease: {:.2f}%".format((no_disease / len(dataset.target) * 100)))
cp_ = pd.get_dummies(dataset['cp'], prefix='cp')

thal_ = pd.get_dummies(dataset['thal'], prefix='thal')

slope_ = pd.get_dummies(dataset['thal'], prefix='thal')
frames = [dataset, cp_, thal_, slope_]

dataset = pd.concat(frames, axis=1)

dataset.head()
dataset = dataset.drop(columns=['cp', 'thal', 'slope'])

dataset.head()
y = dataset.target.values

X = dataset.drop(['target'], axis=1)
# FIX HERE X = (X - np.min(X)) / (np.max(X) - np.min(X)).values

X.head()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from __future__ import absolute_import, division, print_function



import pathlib



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)
def build_model():

    model = keras.Sequential([

        layers.Dense(32, activation=tf.nn.relu, input_shape=[len(x_train.keys())], kernel_initializer='random_normal'),

        layers.Dense(32, activation=tf.nn.relu, kernel_initializer='random_normal'),

        layers.Dense(32, activation=tf.nn.relu, kernel_initializer='random_normal'),

        layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer='random_normal')

    ])



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = build_model()
# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('')

        print('.', end='')

        

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50)



history = model.fit(x_train, y_train, epochs=500, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
eval_model = model.evaluate(x_train, y_train)

print("Model result -> Loss: {:.2f} Accurancy: {:.2f}%".format(eval_model[0], eval_model[1]*100))
import matplotlib.pyplot as plt

%matplotlib inline

def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Accurancy')

    plt.plot(hist['epoch'], hist['acc'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_acc'],

           label = 'Val Error')

    #plt.ylim([0,5])

    plt.legend()



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.plot(hist['epoch'], hist['loss'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_loss'],

           label = 'Val Error')

    #plt.ylim([0,20])

    plt.legend()

    plt.show()
plot_history(history)
y_pred = model.predict(x_test)

y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.suptitle("Confusion Matrix")

plt.figure(figsize=(6, 6))

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)

plt.show()
test_acc = cm[0, 0] + cm[1, 1]

not_test_acc = cm[1, 0] + cm[0, 1]

print("Test accurancy: {:.2f}%".format(100 - (not_test_acc / test_acc)*100))