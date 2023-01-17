""" объект последовательной модели нейронной сети """

from tensorflow.keras.models import Sequential

""" полносвязный слой нейронной сети """

from tensorflow.keras.layers import Dense

""" функции для получения обратной связи во время обучения нейронной сети"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

""" вспомогательный модуль Keras для предварительной обработки данных """

from tensorflow.keras import utils

""" библиотекf для работы с матрицами (многомерными тензорами) и линейной алгеброй """

import numpy as np

""" библиотека для считывания и записи файлов в формате ".csv" и других табличных форматах,

    а также для их быстрой и удобной обработки """

import pandas as pd

""" библиотека для работы с операционной системой """

import os 



"""Зафиксируем генератор случайных чисел. Его не менять!!!"""

from numpy.random import seed

seed(2020)

from tensorflow.random import set_seed

set_seed(2020)

""" библиотека для визуализации данных """

import matplotlib.pyplot as plt

%matplotlib inline 

""" посмотрим, какие файлы храняться в директории """

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
"""функция "read_csv" презназначена для считывания файлов в формате ".csv"

filepath_or_buffer - путь к файлу,

index_col - необязательный параметр, который указывает какую колонку использовать, как индекс.

Если параметр не указан, что pandas создаст столбец с индексами самостоятельно

Подробнее о возможностях функции https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html"""

train_df = pd.read_csv(filepath_or_buffer='/kaggle/input/nnfashinmnist/train.csv', index_col='id')

test_df = pd.read_csv(filepath_or_buffer='/kaggle/input/nnfashinmnist/test.csv', index_col='id')

sample_submission = pd.read_csv(filepath_or_buffer='/kaggle/input/nnfashinmnist/sample_submission.csv', index_col='id')
"""iloc позволяет считать необходимые строки и столбцы, обращаясь к ним не по именам, а по порядковой нумерации

Подробнее про функцию iloc https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html

Запись train_df['label'], извлекает из всей таблицы столбец с названием 'label' 

.values - преобразовывает pandas DataFrame (объект таблицы) в numpy массив"""

x_train, y_train = train_df.iloc[:,:-1].values, train_df['label'].values

x_test = test_df.values
assert x_train.shape[1] == x_test.shape[1], 'Количество признаков в тренировочном и тестовом наборах должно совпадать'
classes = {0:'футболка', 1:'брюки', 2:'свитер', 3:'платье', 4:'пальто',

           5:'туфли', 6:'рубашка', 7:'кроссовки', 8:'сумка', 9:'ботинки'}
plt.figure(figsize=(10,10))

for i in range(100,150):

    plt.subplot(5,10,i-100+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i].reshape((28,28)), cmap=plt.cm.binary)

    plt.xlabel(classes[y_train[i]])
# Векторизованные операции

# Применяются к каждому элементу массива отдельно

x_train = x_train / 255.0

# Все преобразования с тренировочным наборам повторяем и для тестового

x_test = x_test / 255.0
print(y_train[100])
y_train = utils.to_categorical(y_train)
print(y_train[100])
# Создаем последовательную модель

model = Sequential()

# Входной полносвязный слой, 800 нейронов, 784 входа в каждый нейрон

model.add(Dense(units=800, input_dim=784, activation="relu"))

# Выходной полносвязный слой, 10 нейронов (по количеству рукописных цифр)

model.add(Dense(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())
model.fit(x_train, y_train, 

          batch_size=200, 

          epochs=10,  

          verbose=1)
"""Если вы хотите применить подход с обратной связью,

    то снимите комментарии со следующего кода.

    Попробуйте менять параметры monitor и patience"""

# # Создаем последовательную модель

# model = Sequential()

# # Входной полносвязный слой, 800 нейронов, 784 входа в каждый нейрон

# model.add(Dense(units=800, input_dim=784, activation="relu"))

# # Выходной полносвязный слой, 10 нейронов (по количеству рукописных цифр)

# model.add(Dense(units=10, activation="softmax"))

# # компилируем модель

# model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# print(model.summary())

# # создаем лист с обратными связями

# # в случае restore_best_weights=True, применять  ModelCheckpoint не нужно

# callbacks_list = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),

#                   ModelCheckpoint(filepath='my_model.h5',

#                                   monitor='loss',

#                                   save_best_only=True),

#                  ]

# # добавляем лист с обратными связями в параметр callbacks

# model.fit(x_train, y_train,

#             batch_size=200,

#             epochs=10,

#             callbacks=callbacks_list,

#             verbose=1)
predictions = model.predict(x_train)
# Меняйте значение n чтобы просмотреть результаты распознавания других изображений

n = 2020

plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)

plt.show()
print(predictions[n])
"""argmax - находит максимальный элемент массива и возвращает его номер в массиве"""

np.argmax(predictions[n])
"""Передаем в словарь classes номер максимального элемента предсказанного сетью

Словарь возвращает название класса, который скрыт за этим номером"""

classes[np.argmax(predictions[n])]
np.argmax(y_train[n])
classes[np.argmax(y_train[n])]
"""делаем предсказания по всем тестовым данным"""

predictions = model.predict(x_test)

"""извлекаем номера предсказаний с максимальными вероятностями по всем объектам тестового набора"""

predictions = np.argmax(predictions, axis=1)

predictions
"""используем файл с правильным шаблоном формата записи ответов и пишем в него наши предсказания"""

sample_submission['label'] = predictions
"""to_csv - пишет табличные данные в файл '.csv' """

sample_submission.to_csv('sample_submission.csv')