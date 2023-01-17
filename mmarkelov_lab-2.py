import numpy as np 

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras



mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



train_images = train_images / 255.0

test_images = test_images / 255.0



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.imshow(train_images[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape = (28, 28)),

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(10, activation = 'softmax')

])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])



model.fit(train_images, train_labels, epochs = 10)



test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

print('\nТочность на проверочных данных:', test_acc)
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



data = datasets.load_wine()

data_train, data_test, label_train, label_test = train_test_split(data['data'],

                                                                  data['target'],

                                                                  test_size=0.2)



scaler = StandardScaler()

data_train = scaler.fit_transform(data_train)

data_test = scaler.fit_transform(data_test)





model = keras.Sequential([

    keras.layers.Dense(128, activation = 'tanh'),

    keras.layers.Dense(10, activation = 'softmax')

])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])



model.fit(data_train, label_train, epochs = 10)



test_loss, test_acc = model.evaluate(data_test, label_test, verbose = 2)

print('\nТочность на проверочных данных:', test_acc)

data = datasets.load_breast_cancer()

data_train, data_test, label_train, label_test = train_test_split(data['data'],

                                                                  data['target'],

                                                                  test_size=0.2)



scaler = StandardScaler()

data_train = scaler.fit_transform(data_train)

data_test = scaler.fit_transform(data_test)



model = keras.Sequential([

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(2, activation = 'softmax')

])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])



model.fit(data_train, label_train, epochs = 10)



test_loss, test_acc = model.evaluate(data_test, label_test, verbose = 2)

print('\nТочность на проверочных данных:', test_acc)
