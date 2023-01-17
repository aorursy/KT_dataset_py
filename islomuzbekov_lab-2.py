# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras



mnist = keras.datasets.mnist

(train_images, train_labels),  (test_images, test_labels) = mnist.load_data()



# Any results you write to the current directory are saved as output.
print(train_images.shape)

print(test_images.shape)

print(train_labels.shape)

print(test_labels.shape)



train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(train_images[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



wine = load_wine()

# print(wine['DESCR'])



wine_train_data, wine_test_data, train_labels, test_labels = train_test_split(wine['data'],

                                                                  wine['target'],

                                                                  test_size=0.2)



scaler = StandardScaler()

wine_train_data = scaler.fit_transform(wine_train_data)

wine_test_data = scaler.fit_transform(wine_test_data)



model = keras.Sequential([

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(10, activation = 'softmax')

])



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(wine_train_data, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(wine_test_data, test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)
from sklearn.datasets import load_breast_cancer



cancer = load_breast_cancer()

# print(cancer['DESCR'])



cancer_train_data, cancer_test_data, train_labels, test_labels = train_test_split(cancer['data'],

                                                                  cancer['target'],

                                                                  test_size=0.2)



scaler = StandardScaler()

cancer_train_data = scaler.fit_transform(cancer_train_data)

cancer_test_data = scaler.fit_transform(cancer_test_data)



model = keras.Sequential([

    keras.layers.Dense(128, activation = 'sigmoid'),

    keras.layers.Dense(10, activation = 'softmax')

])



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(cancer_train_data, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(cancer_test_data, test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)