import numpy as np

import keras

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

import cv2

from keras.applications.inception_resnet_v2 import InceptionResNetV2
IMG_SIZE = (224, 224)  # размер входного изображения сети
import pandas as pd

my_submission = pd.DataFrame(columns=['id', 'label'])

my_submission.id = np.arange(1,12501).astype(int)

my_submission.label = np.zeros(12500)

my_submission.head(3)
from numpy import random
random.seed(12412)

'''функция для небольшого изменения цвета'''

def change_value(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img_hsv[...,0] = img_hsv[...,0] + random.randint(-40, 40)

    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    img_new = array_to_img(img_rgb) # для того что бы изображение не выходило за границы (0, 255)

    array = img_to_array(img_new)

    return array
import re

from glob import glob



train_files = glob('../input/train/*.jpg')

test_files = glob('../input/test/*.jpg')



random.seed(29)

# загружаем входное изображение и предобрабатываем

def load_image_with_augmentation(path, target_size=IMG_SIZE):

    images = [] # создадим лист для аугментированных изображений

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array_1 = img_to_array(img)

    images.append(array_1) # добавим оригинальное изображение

    array_2 = cv2.flip(array_1, flipCode=1)

    images.append(array_2) # добавим зеркальное изображение

    M = cv2.getRotationMatrix2D(center=(112, 112), angle=random.normal(loc=0.0, scale=10.0), scale=1.3)

    array_3 = cv2.warpAffine(array_1, M, (224, 224))

    images.append(array_3) # добавим повернутое изображение

    array_4 = change_value(array_1)

    images.append(array_4) # добавим изображение с измененным цветом

    a = random.choice((0, 1, 2, 3))

    array = images[a]

    return preprocess_input(array)  # предобработка для VGG16



def load_image(path, target_size=IMG_SIZE):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    array = img_to_array(img)

    return preprocess_input(array)  # предобработка для VGG16



# генератор для последовательного чтения обучающих данных с диска

def fit_generator(files, batch_size=32):

    while True:

        random.shuffle(files)

        for k in range(len(files) // batch_size):

            i = k * batch_size

            j = i + batch_size

            if j > len(files):

                j = - j % len(files)

            x = np.array([load_image_with_augmentation(path) for path in files[i:j]])

            y = np.array([1. if re.search('.*/dog\.\d', path) else 0. for path in files[i:j]])

            yield (x, y)



# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            yield np.array([load_image(path)])
len(train_files)
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, path in enumerate(test_files[:5], 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%s' % path.split('/')[-1]);
# base_model -  объект класса keras.models.Model (Functional Model)

# base_model = VGG16(include_top = False,

#                    weights = 'imagenet',

#                    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
base_model = ResNet50(include_top = False,

                      weights = 'imagenet',

                      input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
# base_model = InceptionResNetV2(include_top = False,

#                                weights = 'imagenet',

#                                input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
len(base_model.layers)
for layer in base_model.layers[:-25]:

    layer.trainable = False
# фиксируем веса до 6-ого слоя

# for num, layer in enumerate(base_model.layers):

#     if num <= 6:

#         layer.trainable = False

#     else:

#         layer.trainable = True
# base_model.summary()
from keras.layers.normalization import BatchNormalization
seed = 1854241

kernek_initializer = keras.initializers.glorot_normal(seed=seed)

bias_initializer = keras.initializers.normal(stddev=1., seed=seed)

act = keras.layers.PReLU()



x = base_model.layers[-5].output

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(24, activation=act, kernel_regularizer=keras.regularizers.l1(1e-4),

                       bias_initializer=bias_initializer, kernel_initializer=kernek_initializer)(x)

x = BatchNormalization()(x)

x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Dense(24, activation=act, kernel_regularizer=keras.regularizers.l1(1e-4), 

                       bias_initializer=bias_initializer, kernel_initializer=kernek_initializer)(x)

x = BatchNormalization()(x)

x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l1(1e-4), 

                       bias_initializer=bias_initializer, kernel_initializer=kernek_initializer)(x)

model = Model(inputs=base_model.input, outputs=x)
# model.summary()
model.evaluate



optimizer_ = keras.optimizers.adam(lr=3e-4)

model.compile(optimizer=optimizer_, 

              loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss)

              metrics=['accuracy'])
def scheduler(epoch):

    if epoch < 180:

        return 3e-4

    elif 180 <= epoch < 250:

        return 1e-4

    elif 250 <= epoch < 330:

        return 5e-5

    else:

        return 1e-5



callback = keras.callbacks.LearningRateScheduler(scheduler)
random.seed(29)

random.shuffle(train_files)  # перемешиваем обучающую выборку



train_val_split = 100  # число изображений в валидационной выборке



validation_data = next(fit_generator(train_files[:train_val_split], train_val_split))



# запускаем процесс обучения

model.fit_generator(fit_generator(train_files[train_val_split:]),  # данные читаем функцией-генератором

        steps_per_epoch=30,  # число вызовов генератора за эпоху

        epochs=450,  # число эпох обучения

        validation_data=validation_data)#, callbacks=[callback])
pred = model.predict_generator(predict_generator(test_files), len(test_files), max_queue_size=500)
%matplotlib inline

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 20))

for i, (path, score) in enumerate(zip(test_files[160:][:10], pred[160:][:10]), 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%.3f' % score);
with open('submit_2.csv', 'w') as dst:

    dst.write('id,label\n')

    for path, score in zip(test_files, pred):

        dst.write('%s,%f\n' % (re.search('(\d+)', path).group(0), score))
my_submission.label = pred

my_submission.head()
# my_submission.to_csv('submit.csv', index=False)