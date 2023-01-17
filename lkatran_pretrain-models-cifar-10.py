import numpy as np

# наборы данных для экспериментов

from tensorflow.keras.datasets import cifar10

import tensorflow as tf

## слои

# последовательная модель (стек слоев)

from tensorflow.keras.models import Sequential, Model

# полносвязный слой и слой выпрямляющий матрицу в вектор

from tensorflow.keras.layers import Dense, Flatten

# слой выключения нейронов и слой нормализации выходных данных (нормализует данные в пределах текущей выборки)

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

# слои свертки и подвыборки

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D



## callbacks и вспомогательные функции

# работа с обратной связью от обучающейся нейронной сети

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# вспомогательные инструменты

from tensorflow.keras import utils

from tensorflow.keras.regularizers import *

from tensorflow.keras.preprocessing import image



## предварительно обученные нейронные сети и их вспомогательные инструменты

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from tensorflow.keras.applications import ResNet50, InceptionV3, DenseNet201, EfficientNetB5, InceptionResNetV2, Xception, NASNetLarge, ResNet152V2



import os

from tensorflow.random import set_seed

def seed_everything(seed):

    np.random.seed(seed)

    set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 42

seed_everything(seed)



# работа с изображениями

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
size = 75

X_test = tf.image.resize(X_test, [size,size])

X_train = tf.image.resize(X_train, [size,size])
X_train.shape
# Размер мини-выборки

batch_size = 128

# Количество классов изображений

nb_classes = 10

# Количество эпох для обучения

nb_epoch = 25

# Размер изображений

img_rows, img_cols = X_train.shape[1], X_train.shape[2]

# Количество каналов в изображении: RGB

img_channels = X_train.shape[3]

# Названия классов из набора данных

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
X_train /= 255

X_test /= 255
Y_train = utils.to_categorical(y_train, nb_classes)

Y_test = utils.to_categorical(y_test, nb_classes)
n = 101

plt.imshow(X_train[n].numpy())

plt.show()

print("Номер класса:", y_train[n][0])

print("Тип объекта:", y_train[n][0])
# функция управляющая изменениями шага обучения в процессе тренировки нейронной сети

LR_START = 0.00001

LR_MAX = 0.0001

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 2

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



# построим график изменения шага обучение в зависимости от эпох

rng = [i for i in range(nb_epoch)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
%%time

def get_model(use_model):

    # загружаем веса предварительно обученной нейронной сети

    base_model = use_model(weights='imagenet', 

                      include_top=False, 

                      input_shape=(img_rows, img_cols, img_channels))

    base_model.trainable = True

    x = base_model.output

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    x = BatchNormalization()(x)

    x = GaussianDropout(0.8)(x)

    predictions = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)

# создаем модель

model = get_model(ResNet50)
# model.summary()
callbacks_list = [EarlyStopping(monitor='val_loss', patience=10),

                  ModelCheckpoint(filepath='my_model.h5',

                                  monitor='val_loss',

                                  save_best_only=True),

                  lr_callback

                  ]# ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3) 

# optimizer = tf.keras.optimizers.Adam() # learning_rate=0.00001

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
%%time

history = model.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=nb_epoch,

              callbacks=callbacks_list,

              validation_split=0.1,

              verbose=1)
# Оцениваем качество обучения модели на тестовых данных

scores = model.evaluate(X_test, Y_test, verbose=1)

print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
plt.plot(history.history['accuracy'], 

         label='Доля правильных ответов на обучающем наборе')

plt.plot(history.history['val_accuracy'], 

         label='Доля правильных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля правильных ответов')

plt.legend()

plt.show()
plt.plot(history.history['loss'], 

         label='Оценка потерь на обучающем наборе')

plt.plot(history.history['val_loss'], 

         label='Оценка потерь на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Оценка потерь')

plt.legend()

plt.show()