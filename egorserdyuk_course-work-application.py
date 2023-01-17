import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.datasets import mnist # Получаем MNIST набор данных

from keras.models import Model # Базовый класс для настройки и обучения нейронной сети

from keras.layers import Input, Dense # Два типа слоя, которые мы будем использовать

from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
batch_size = 128 # в каждой итерации мы одновременно рассматриваем 128 обучающих примеров

num_epochs = 20 # мы повторяем двадцать раз за весь тренировочный набор

hidden_size = 512 # в обоих скрытых слоях будет 512 нейронов
num_train = 60000 # возьмем из MNIST 60000 примеров обучения

num_test = 10000 # возьмем из MNIST 10000 тестовых примеров



height, width, depth = 28, 28, 1 # кодируем изображения MNIST в размер 28x28 и оттенки серого

num_classes = 10 # 10 классов, по одному на цифру



(X_train, y_train), (X_test, y_test) = mnist.load_data() # загружаем данные MNIST



X_train = X_train.reshape(num_train, height * width) # Сводим данные в одномерный массив

X_test = X_test.reshape(num_test, height * width) # Сводим данные в одномерный массив

X_train = X_train.astype('float32') 

X_test = X_test.astype('float32')

X_train /= 255 # Нормализуем данные в диапазон от 0 до 1

X_test /= 255 # Так же нормализуем данные в диапазон от 0 до 1



Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels

Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
inp = Input(shape=(height * width,)) # Ввод - одномерный вектор, размера 784

hidden_1 = Dense(hidden_size, activation='relu')(inp) # Первый скрытый слой ReLU

hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Второй скрытый слой ReLU

out = Dense(num_classes, activation='softmax')(hidden_2) # Выходной слой softmax



model = Model(input=inp, output=out) # Для определения модели просто указываем входной и выходной слои
model.compile(loss='categorical_crossentropy', # используем функцию потерь cross-entropy loss

              optimizer='adam', # используем Adam optimiser

              metrics=['accuracy']) # сообщаем точность
model.fit(X_train, Y_train, # Тренируем модель

          batch_size=batch_size, nb_epoch=num_epochs,

          verbose=1, validation_split=0.1) # 10% данных используем для проверки

model.evaluate(X_test, Y_test, verbose=1) # Оценка точности обученной модели на тестовом наборе данных