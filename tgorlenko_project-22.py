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
import os

import shutil

import re

import math



import pandas as pd

import numpy as np



import PIL.Image



from random import shuffle

from glob import glob



from sklearn.model_selection import train_test_split



#from tensorflow.python.keras.applications import VGG16



from keras.applications import VGG16



from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



from tensorflow import keras

from tensorflow.keras.models import Model

#from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array



from keras import models

from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint



from keras.optimizers import Adam
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head(10)
df.diagnosis.value_counts()
image_train, image_test, y_train, y_test = train_test_split(np.array(df.path), 

                                                            np.array(df.diagnosis), 

                                                            test_size=0.3,

                                                            random_state=123, 

                                                            stratify=df.diagnosis)
image_train
image_and_class_train = dict(zip(image_train, y_train))

image_and_class_test = dict(zip(image_test, y_test))
IMG_SIZE = (224, 224)  # размер входного изображения сети

NUM_CLASSES = 5        # число классов
# загружаем входное изображение и предобрабатываем

def load_image(path, target_size=IMG_SIZE):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array = img_to_array(img)

    return preprocess_input(array)  # предобработка для VGG16
# генератор для последовательного чтения обучающих данных с диска

def fit_generator(files, batch_size=32):

    while True:

        shuffle(files)

        for k in range(math.ceil(len(files) / batch_size)):   # округляем до ближайшего целого вверх

            i = k * batch_size                                # k -- номер батча в проходе                      

            j = i + batch_size

            if j > len(files):

                j = len(files)

            x = np.array([load_image(path) for path in files[i:j]])         # картинки в виде матрицы

            label = np.array([image_and_class_train[path] for path in files[i:j]])   # метки классов

            y = keras.utils.to_categorical(label, num_classes=NUM_CLASSES)      # one hot кодирование

            yield (x, y)
# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            yield np.array([load_image(path)])
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, path in enumerate(image_train[:10], 1):

    subplot = fig.add_subplot(math.ceil(i / 5), 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('{} \n label: {}'.format(os.path.basename(path), image_and_class_train[path]))
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# фиксируем все веса предобученной сети кроме последнего блока 

set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv3':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
conv_base.summary()
model_2 = models.Sequential()

model_2.add(conv_base)  # кусок VGG-16 добавлен в модель

model_2.add(layers.BatchNormalization())

model_2.add(layers.Flatten())

#model_1.add(layers.Dense(512, activation='relu'))

model_2.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model_2.summary()
model_2.compile(optimizer=Adam(lr=0.005), 

              loss='categorical_crossentropy',  # функция потерь 'categorical_crossentropy' (log loss

              metrics=['accuracy'])
shuffle(image_train)  # перемешиваем обучающую выборку



train_val_split = 100  # число изображений в валидационной выборке



validation_data = next(fit_generator(image_train[:train_val_split], train_val_split))



# запускаем процесс обучения

history = model_2.fit_generator(fit_generator(image_train[train_val_split:]),  # данные читаем функцией-генератором

        steps_per_epoch=10,  # число вызовов генератора за эпоху

        epochs=100,  # число эпох обучения

        validation_data=validation_data

#         callbacks=[EarlyStopping(patience = 5),

#                    ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',

#                                   verbose=1,

#                                   save_best_only=True)]

                               )
start = 0

plt.plot(history.history['loss'][start:])

plt.plot(history.history['val_loss'][start:])

plt.legend(['Train loss', 'Validation loss'])
plt.plot(history.history['acc'][start:])

plt.plot(history.history['val_acc'][start:])

plt.legend(['Train acc', 'Validation acc'])
# загружаем веса модели для наименьшего loss 

#model_2.load_weights('weights.10-3.12.hdf5')
pred = model_2.predict_generator(predict_generator(image_test), len(image_test), max_queue_size=500)
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, (path, score) in enumerate(zip(image_test[70:][:10], pred[70:][:10]), 1):

    subplot = fig.add_subplot(math.ceil(i / 5), 5, i)

    plt.imshow(plt.imread(path))

    subplot.set_title('label: {} \n prediction: {} \n model confidence: {:.3f}'\

                      .format(image_and_class_test[path],

                              int(np.argmax(score)),

                             np.max(score)))