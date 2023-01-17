# Магическое слово для отрисовки их в юпитер ноутбуке

%matplotlib inline



# Для повторяемости результатов

RANDOM_SEED = 42



# Для нейронок

import keras

import numpy as np

import pandas as pd

import matplotlib as plot

import matplotlib.pyplot as plt

import tensorflow as tf

import random

import torch

import torchvision

from sklearn.model_selection import * 

from catboost import *

from keras import *

from keras.layers.convolutional import Conv2D

from keras.layers import *

from tensorflow.nn import *

from keras.callbacks import *

from keras.models import *

from keras.optimizers import *

from keras.preprocessing import image

from sklearn.metrics import accuracy_score



# Фиксируем рэндом сид для повторяемости результатов

np.random.seed(RANDOM_SEED)

tf.set_random_seed(RANDOM_SEED)
# Загружаем датасет MNIST

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()



# Скалируем/Нормализуем данные (они были от 0 до 255, а стали от 0 до 1)

X_train = X_train.astype(float) / 255.

X_test = X_test.astype(float) / 255.
# Поменяйте номер ниже, чтобы посмотреть на различные примеры из тестового датасета

picture_num = 1337



plt.imshow(X_test[picture_num], cmap="Greys"); 

print(y_test[picture_num])
def flatten_images(x):

    # Функция делает из прямоугольных изображений плоские, вытянутые в вектор длиной 28*28 = 784

    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])



# "Плющим" все датасеты

X_train = flatten_images(X_train)

X_test = flatten_images(X_test)



def one_hot_labels(y):

    # Функция делает из ответов вида 1, 4, 7, 9 их one hot представление для подачи в нейронку

    result = np.zeros((y.size, 10))

    result[np.arange(y.size), y] = 1

    return result



# "Ванхотим" все ответы 

y_train_one_hot = one_hot_labels(y_train)

y_test_one_hot = one_hot_labels(y_test)
# Импортируем заготовку для нашей сетки

from keras.models import Sequential

# И полносвязные слои

from keras.layers import Dense



# Создаем пустую сетку

net = Sequential()



# Наполняем ее внутренними слоями

net.add(Dense(128, activation = relu, input_dim = 28*28))

net.add(Dense(512, activation = relu)) 

net.add(Dropout(rate = 0.2))

net.add(Dense(10, activation = softmax)) 



# # Наполняем ее 3 внутренними слоями

# net.add(Dense(512, activation='relu', input_dim=28*28)) 

# net.add(Dense(128, activation='relu')) 

# net.add(Dense(32, activation='relu')) 

# 

# # + 1 выходной с активацией софтмакс

# net.add(Dense(10, activation='softmax')) 



# Собираем в кучу нашу модель, указываем ей ошибку для минимизации (категориальная кроссэнтропия)

# Также выбираем оптимизатор (любимый Адам) и метрику, которая будет показываться (точность)

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Показываем какие у нас слои есть и сколько в них весов

net.summary()
NoNaN = TerminateOnNaN()

earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode = 'auto')

mcp_save = ModelCheckpoint('weights.best.hdf5', save_best_only = True, monitor = 'val_loss', mode = 'auto')

lr_loss = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 7, verbose = 1, mode = 'auto')

rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
# Тренируем сетку

net.fit(X_train, y_train_one_hot, batch_size = 128, epochs = 12, verbose = 2, callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction], validation_data = (X_test, y_test_one_hot))



net.load_weights("weights.best.hdf5")



net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = net.fit(X_train, y_train_one_hot, batch_size = 128, epochs = 12, verbose = 2, callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction], validation_data = (X_test, y_test_one_hot))
# Делаем предикт сеткой и получаем One Hot матрицу

y_pred_one_hot = net.predict(X_test)

# Преобразовываем ее в ответы

y_pred_labels = np.argmax(y_pred_one_hot, axis=1)



# Считаем ваш скор

your_score = accuracy_score(y_test, y_pred_labels)

print('Your accuracy on a test dataset =', your_score)