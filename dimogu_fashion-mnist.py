# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')

#изначальные размеры изображений

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

#вытаскиваем векторы изображений

X = np.array(data_train.iloc[:,1:])

#вытаскиваем вектор меток

y = to_categorical(np.array(data_train.iloc[:, 0]))

#разделение на тренировочную и проверочную выборку

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Тестовые вектора

X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))

#нормализация данных

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
#честно взятая архитектура с fashion mnist 

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

#задание изначальных параметров: размера батча, количества классов, эпох обучения

batch_size = 256

num_classes = 10

epochs = 50



#измерения входных данных

img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 #задания распределения начальных весов системы

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

#сжатие

model.add(MaxPooling2D((2, 2)))

#снижение ошибки переобучения

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

#компиляция модели

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

#вывод модели

model.summary()
#обучения модели

history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)

print("score")

print(score[1])
#много раз использованная проверка графиком обучения, на котором наглядно показаны между "эмпирическими" и "теоретическими" предсказаниями

# первая точка оторвана из-за ненатренированных весов

# если кто-то писал в MatLab синтаксис может показаться похожим

import matplotlib.pyplot as plt

%matplotlib inline

accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()


#предсказания полученные на тренированной модели на "тестовой" выборке

predicted_classes = model.predict_classes(X_test)



#получение индексов для построения статистики

y_true = data_test.iloc[:, 0]

correct = np.nonzero(predicted_classes==y_true)[0]

incorrect = np.nonzero(predicted_classes!=y_true)[0]

# точность обученной модели по классам и в целом

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))