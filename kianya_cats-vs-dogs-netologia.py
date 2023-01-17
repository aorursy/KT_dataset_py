! ls /.keras/models
from keras import backend as K
import numpy as np

import keras

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import load_img, img_to_array

import os



from sklearn.metrics import log_loss
IMG_SIZE = (224, 224)  # размер входного изображения сети
print(len(os.listdir("../input/train")))
import re

from random import shuffle

from glob import glob



TRAIN_DIR = '../input/train'

TEST_DIR = '../input/test'



train_files = glob("../input/train/*.jpg")

test_files = glob('../input/test/*.jpg')



# загружаем входное изображение и предобрабатываем

def load_image(path, target_size=IMG_SIZE):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array = img_to_array(img)

    return preprocess_input(array)  # предобработка для VGG16



# генератор для последовательного чтения обучающих данных с диска

def fit_generator(files, batch_size=32):

    while True:

        shuffle(files)

        for k in range(len(files) // batch_size):

            i = k * batch_size

            j = i + batch_size

            if j > len(files):

                j = - j % len(files)

            x = np.array([load_image(path) for path in files[i:j]])

            y = np.array([1. if re.match('.*/dog\.\d', path) else 0. for path in files[i:j]])

            yield (x, y)



# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            yield np.array([load_image(path)])
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, path in enumerate(train_files[:10], 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%s' % path.split('/')[-1]);
# base_model -  объект класса keras.models.Model (Functional Model)

base_model = VGG16(include_top = False,

                   weights = 'imagenet',

                   input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
# фиксируем все веса предобученной сети

for layer in base_model.layers:

    layer.trainable = False
base_model.summary()
x = base_model.layers[-5].output

x = keras.layers.Flatten()(x)

x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(32,  # один выход

                activation='sigmoid',  # функция активации  

                #kernel_regularizer=keras.regularizers.l1(1e-4)

                      )(x)

x = keras.layers.Dense(32,  # один выход

                activation='sigmoid',  # функция активации  

                #kernel_regularizer=keras.regularizers.l1(1e-4)

                      )(x)

x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Dense(1,  # один выход

                activation='sigmoid',  # функция активации  

                kernel_regularizer=keras.regularizers.l2(1e-4)

                      )(x)

model = Model(inputs=base_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', 

              loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss

              metrics=['accuracy',  ])
shuffle(train_files)  # перемешиваем обучающую выборку



train_val_split = 100  # число изображений в валидационной выборке



validation_data = next(fit_generator(train_files[:train_val_split], train_val_split))



# запускаем процесс обучения

model.fit_generator(fit_generator(train_files[train_val_split:]),  # данные читаем функцией-генератором

        steps_per_epoch=10,  # число вызовов генератора за эпоху

        epochs=15,  # число эпох обучения

        validation_data=validation_data)
model.save('model')
pred = model.predict_generator(predict_generator(test_files), len(test_files), max_queue_size=500)
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, (path, score) in enumerate(zip(test_files[80:][:10], pred[80:][:10]), 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%.3f' % score);
# Рассмотрим метрики для анализа полученной модели
with open('submit.txt', 'w') as dst:

    dst.write('id,label\n')

    for path, score in zip(test_files, pred):

        dst.write('%s,%f\n' % (re.search('(\d+)', path).group(0), score))
# LogLoss = 1.04979