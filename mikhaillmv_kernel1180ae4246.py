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
from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

# Устанавливаем seed для повторяемости результатов

np.random.seed(42)

# Загружаем данные

data = np.load('/kaggle/input/mnist-lmv/mnist.npz')

X_train, y_train, X_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

# Размер изображения

img_rows, img_cols = 28, 28

# Преобразование размерности изображений

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

# Нормализация данных

X_train = X_train.astype('float32')

X_train /= 255

# Преобразуем метки в категории

Y_train = np_utils.to_categorical(y_train, 10)

X_train.shape
from keras.layers import Dropout, Flatten, MaxPool2D, Conv2D

# Создаем последовательную модель https://keras.io/models/sequential/

model = Sequential()

# Добавляем слой

model.add( Conv2D(filters=8, kernel_size=(5,5), activation="relu", input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add( Dropout(rate=0.1, seed=42) )

model.add( Conv2D(filters=16, kernel_size=(3,3), activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add( Dropout(rate=0.1, seed=42) )

model.add(Flatten())

model.add( Dense(10, input_dim=484, activation="softmax") )

# Компилируем модель

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Обучаем сеть

model.fit(X_train, Y_train, batch_size=200, epochs=20, validation_split=0.2, verbose=1)
# Аналогичная предобработка дянных для набора X_test

# Преобразование размерности изображений

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# Нормализация данных

X_test = X_test.astype('float32')

X_test /= 255

# Преобразуем метки в категории

Y_test = np_utils.to_categorical(y_test, 10)
# Оцениваем качество обучения сети на тестовых данных

predict = model.predict(X_test)

# Преобразуем категории в метки классов 0-9

predict = np.argmax(predict, axis=1)

print(predict)

predict_cat = np_utils.to_categorical(predict, 10)

from keras.metrics import categorical_accuracy

categorical_accuracy(Y_test, predict_cat)
accuracy = categorical_accuracy(Y_test, predict_cat)

sum_y_test = 0.0

for i in range(accuracy.shape[0]):

    sum_y_test += accuracy[i]

print(sum_y_test / accuracy.shape[0])
# Сохраняем сеть для последующего использования

# Генерируем описание модели в формате json

model_json = model.to_json()

json_file = open("mnist_model.json", "w")

# Записываем архитектуру сети в файл

json_file.write(model_json)

json_file.close()

# Записываем данные о весах в файл

model.save_weights("mnist_model.h5")
# Загружаем данные об архитектуре сети

from keras.models import model_from_json

json_file = open("mnist_model.json", "r")

loaded_model_json = json_file.read()

json_file.close()

# Создаем модель

loaded_model = model_from_json(loaded_model_json)

# Загружаем сохраненные веса в модель

loaded_model.load_weights("mnist_model.h5")