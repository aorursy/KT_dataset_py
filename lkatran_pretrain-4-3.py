"""Вспомогательные библиотеки"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 



"""Предварительно обученные нейронные сети"""

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications import ResNet50, InceptionV3, DenseNet201, Xception

"""Предварительная обработка для каждой архитектуры"""

from tensorflow.keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess

from tensorflow.keras.applications.xception import preprocess_input as Xception_preprocess

from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_ResNet50_preprocess

from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess

from tensorflow.keras.layers.experimental import preprocessing







"""Зафиксируем генератор случайных чисел. Его не менять!!!"""

from numpy.random import seed

seed(2020)

from tensorflow.random import set_seed

set_seed(2020)



import os

""" посмотрим, какие файлы храняться в директории """

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Названия классов из набора данных

import json

with open('/kaggle/input/pretrain-urfu/classes.json', 'r') as file:

    classes = json.load(file)
"""Так как данные храняться не в формате таблиц, а в формате многомерных тензоров numpy,

то применим для загрузки данных функцию numpy load()"""

X_test = np.load('/kaggle/input/pretrain-urfu/test.npy')
plt.figure(figsize=(20,20))

for i in range(100,109):

    plt.subplot(3,3,i-100+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_test[i])
X_test = X_test * 255

X_test = X_test.astype('float32')

"""Используйте соответствующий препроцессинг для каждой архитектуры

В случае работы с DenseNet201, закомментируйте следующую строку"""

X_test = VGG16_ResNet50_preprocess(X_test)



"""Для архитектуры DenseNet201 раскомментируйте строки ниже"""

# layer = preprocessing.Normalization()

# layer.adapt(X_test)

# X_test = layer(X_test)
"""Тут необходимо менять архитектуру при каждом новом эксперименте,

чтобы оценить новую модель"""

model = VGG16(include_top=True, weights='imagenet', classes=1000)
model.summary()
"""делаем предсказания по всем тестовым данным"""

predictions = model.predict(X_test)

"""извлекаем номера предсказаний с максимальными вероятностями по всем объектам тестового набора"""

predictions = np.argmax(predictions, axis=1)

predictions
"""используем файл с правильным шаблоном формата записи ответов и пишем в него наши предсказания"""

sample_submission = pd.read_csv('/kaggle/input/pretrain-urfu/sample_submission.csv')

sample_submission['label'] = predictions
"""to_csv - пишет табличные данные в файл '.csv' """

sample_submission.to_csv('sample_submission.csv', index=False)