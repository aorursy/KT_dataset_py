import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, Dense, Flatten

from tensorflow.keras.models import Model

from shutil import copyfile, rmtree

from timeit import default_timer as timer
# Вспомогательная функция для доступа к файлам относительно корня директория с данными.

INPUT_ROOT = "../input/gtsrb-german-traffic-sign"

def from_input(path):

    return os.path.join(INPUT_ROOT, path)
# Загружаем таблицу с данными о данных.

train_info = pd.read_csv(from_input("Train.csv"))

train_info.head()
# Посмотрим как выглядят наши данные.

train_info.describe()
# сколько примеров в каждом из классов

train_info.groupby('ClassId')['ClassId'].count()
test_info =  pd.read_csv(from_input("Test.csv"))

test_info.head()
test_info.describe()
# сколько примеров в каждом из классов

test_info.groupby('ClassId')['ClassId'].count()
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Показываем изображения в сетке 6х8.

nrows = 8

ncols = 6



pic_offset = 0 # Чтобы итерировать по изображениям каждый раз когда запустим код ниже.
def show_images(offset):

    fig = plt.gcf()

    fig.set_size_inches(ncols*3, nrows*3)



    for i in range(43):

        # subplot индексы начинаются с 1

        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('Off')

        subdir = os.path.join(from_input('train'), str(i))

        files = os.listdir(subdir)

        img_path = os.path.join(subdir, files[offset % len(files)])

        img = mpimg.imread(img_path)

        #print(img.shape)

        plt.imshow(img)



    plt.show()
show_images(pic_offset)

pic_offset += 1
TARGET_SIZE = (40, 40) # изображения будут изменены до этого размера

BATCH_SIZE=300
paths = train_info['Path'].values

y_train = train_info['ClassId'].values



indices = np.arange(y_train.shape[0])

randgen = random.Random(62)

randgen.shuffle(indices)



paths = paths[indices]

y_train = y_train[indices]

y_train = to_categorical(y_train, 43)



data=[]



for i, f in enumerate(paths):

    print('\rLoading data {0:.1f}%...'.format((i / len(paths)) * 100), end = '\r')

    image = Image.open(os.path.join(from_input('train'), f.replace('Train/', '')))

    resized_image = image.resize(TARGET_SIZE)

    data.append(np.array(resized_image))



X_train = np.array(data).astype('float32') / 255.0



print('Data loaded.              ')



train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train,

                                    y_train,

                                    batch_size=BATCH_SIZE,

                                    shuffle=True,

                                    seed=17)
paths = test_info['Path'].values

y_test = test_info['ClassId'].values

y_test = to_categorical(y_test, 43)



data=[]



for i, f in enumerate(paths):

    print('\rLoading data {0:.1f}%...'.format((i / len(paths)) * 100), end = '\r')

    image = Image.open(os.path.join(from_input('test'), f.replace('Test/', '')))

    resized_image = image.resize(TARGET_SIZE)

    data.append(np.array(resized_image))



print('Data loaded.              ')



X_test = np.array(data).astype('float32') / 255.0 



test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(X_test,

                                    y_test,

                                    batch_size=BATCH_SIZE,

                                    shuffle=False,

                                    seed=17)
def plot(history):

    %matplotlib inline



    import matplotlib.image  as mpimg

    import matplotlib.pyplot as plt



    acc=history.history['acc']

    loss=history.history['loss']

    epochs=range(len(acc))



    plt.plot(epochs, acc, 'r', "Training Accuracy")

    plt.title('Training accuracy')

    plt.xlabel('Epoch')

    plt.figure()



    plt.plot(epochs, loss, 'r', "Training Loss")

    plt.xlabel('Epoch')

    plt.title('Training loss')
def show_layers(model):

    print('Name\tOutput shape\tActivation\tInitializer')

    for l in model.layers:

        print('{0}({1})\t{2}\t{3}\t{4}'

            .format(l.name,

              l.__class__.__name__,

              l.output_shape,

              l.activation.__name__ if hasattr(l, 'activation') else '<none>',

              l.kernel_initializer.__class__.__name__ if hasattr(l, 'kernel_initializer') else '<none>'))





def custom_summary(model):

    model.summary()

    show_layers(model)
VERBOSE=0
def train_model(model, kernel_initializer, optimizer, epochs):

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



    start_time = timer()

    history = model.fit_generator(train_generator,

                        epochs=epochs,

                        verbose=VERBOSE,

                        steps_per_epoch= round(X_train.shape[0] / BATCH_SIZE))

    end_time = timer()

    

    custom_summary(model)

    print('==============================')

    print('Initializer: ', kernel_initializer)

    print('Optimizer: ', optimizer.__class__.__name__)

    print('Learning rate: ', optimizer.get_config()['learning_rate'])

    print('Epochs: ', epochs)

    print('==============================')

    print('Trained in {0:.2f} minutes'.format((end_time - start_time) / 60))

    

    acc=history.history['acc'][-1]

    test_acc = model.evaluate_generator(test_generator)[1]

    

    print('Results at the end of training: acc={1:.02f}%, test_acc={2:.02f}%'

          .format(i, acc*100, test_acc*100))



    plot(history)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(256, (5, 5), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(256, (7, 7), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)
kernel_initializer='glorot_uniform'

optimizer=Adam(learning_rate=0.0005)

epochs=20



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(256, (7, 7), activation='relu', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='tanh', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(1024, (3, 3), activation='tanh', input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(43, activation='softmax')

])



train_model(model, kernel_initializer, optimizer, epochs)