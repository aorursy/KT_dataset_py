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
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
test.head(3)
train.head(3)
# Получаем метки значений как массив y_train



y_train = train['label'].values
# Удаляем колонку со значениями чтобы преобразовать данные в массив



x_train = train.drop(columns=['label'])

x_test = test.drop(columns=['id'])



x_train = x_train.to_numpy(copy = True)

x_test = x_test.to_numpy(copy = True)
# Изменение формы массивов для нормальной работы с API Keras



x_train = x_train.reshape(len(x_train), 28, 28, 1)

x_test = x_test.reshape(len(x_test), 28, 28, 1)
# Убедиться, что значения являются плавающими

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# Нормализация кодов RGB путем деления его на максимальное значение RGB.

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print('Number of images in x_train', x_train.shape[0])

print('Number of images in x_test', x_test.shape[0])
# Импорт необходимых Keras модулей, содержащих модель и слои

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
input_shape = (28, 28, 1)

# Создание последовательной модели и доавление слоев

model = Sequential()



# Добавление слоя извлечения признаков

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

# Добавление слоя объединения

model.add(MaxPooling2D(pool_size=(2, 2)))

# Добавление слоя сглаживания двумерных массивов в одномерный перед построением полностью связанных слоев

model.add(Flatten())

# Добавление плотного слоя

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=5)
image_index = 4423

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())
sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
sub.head(5)
d = {i: model.predict(x_test[i].reshape(1, 28, 28, 1)).argmax() for i in test['id']}
(pd.DataFrame.from_dict(data=d, orient='index')

   .to_csv('submission.csv', header=False))