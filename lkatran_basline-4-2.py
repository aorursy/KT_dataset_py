import numpy as np

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline 

import os

""" посмотрим, какие файлы храняться в директории """

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Названия классов из набора данных CIFAR-10

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
"""Так как данные храняться не в формате таблиц, а в формате многомерных тензоров numpy,

то применим для загрузки данных функцию numpy load()"""

X_train = np.load('/kaggle/input/cnn-urfu-cifar10/train.npy')

Y_train = np.load('/kaggle/input/cnn-urfu-cifar10/train_label.npy')

X_test = np.load('/kaggle/input/cnn-urfu-cifar10/test.npy')

X_train.shape, Y_train.shape, X_test.shape
plt.figure(figsize=(10,10))

for i in range(100,150):

    plt.subplot(5,10,i-100+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i])

    plt.xlabel(classes[Y_train[i][0]])
x_train = X_train / 255

x_test = X_test / 255
y_train = utils.to_categorical(Y_train, 10)
# Создаем последовательную модель

model = Sequential()

# Первый сверточный слой

model.add(Conv2D(32, (3, 3), padding='same',

                        input_shape=(32, 32, 3), activation='relu'))

# Второй сверточный слой

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# Первый слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout

model.add(Dropout(0.1))



# Третий сверточный слой

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout

model.add(Dropout(0.1))



# Слой преобразования данных из 2D представления в плоское

model.add(Flatten())

# Полносвязный слой для классификации

model.add(Dense(512, activation='relu'))

# Слой регуляризации Dropout

model.add(Dropout(0.3))

# Выходной полносвязный слой

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

              batch_size=128,

              epochs=20,

              validation_split=0.1,

              verbose=2)
plt.plot(history.history['accuracy'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history.history['val_accuracy'], 

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()
plt.plot(history.history['loss'], 

         label='Ошибка на обучающем наборе')

plt.plot(history.history['val_loss'], 

         label='Ошибка на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Ошибка')

plt.legend()

plt.show()
"""делаем предсказания по всем тестовым данным"""

predictions = model.predict(x_test)

"""извлекаем номера предсказаний с максимальными вероятностями по всем объектам тестового набора"""

predictions = np.argmax(predictions, axis=1)

predictions
"""используем файл с правильным шаблоном формата записи ответов и пишем в него наши предсказания"""

sample_submission = pd.read_csv('/kaggle/input/cnn-urfu-cifar10/sample_submission.csv')

sample_submission['label'] = predictions
"""to_csv - пишет табличные данные в файл '.csv' """

sample_submission.to_csv('sample_submission.csv', index=False)