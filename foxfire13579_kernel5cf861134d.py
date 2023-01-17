import sys

import os

import ssl

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np  # linear algebra

import tensorflow as tf



ssl._create_default_https_context = ssl._create_unverified_context

from skimage import io, filters, color, segmentation

from skimage.transform import rescale, resize, downscale_local_mean, rotate

from keras import losses, models, optimizers

from keras.models import Sequential

from keras.layers import (Convolution2D, Dense, Dropout, GlobalAveragePooling2D,

                          GlobalMaxPool2D, Input, MaxPool2D, concatenate, Activation,

                          MaxPooling2D, Flatten, BatchNormalization, Conv2D, AveragePooling2D)

from keras.models import load_model

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

from keras.metrics import binary_accuracy

from sklearn.model_selection import StratifiedShuffleSplit

from keras import regularizers



print(os.listdir("../input/C-NMC_training_data"))
fftn = 150 # размерность

fftm = 150



# подготовка данных, считывание меток: больные - 'all' или 1, здоровые - 'hem' или 0

def preproc_resize(img):

    #print('preproc_resize')

    new_img = img

    new_img = resize(img, (fftn, fftm, 3), anti_aliasing=True)

    return new_img



path = "../input/C-NMC_training_data/"

image_list = []

label = []

path_list = [path + 'fold_0', path + 'fold_1', path + 'fold_2']

for i in path_list:

    dirr_all = i + '/' + 'all'

    dirr_hem = i + '/' + 'hem'

    listdir_all = os.listdir(dirr_all)

    listdir_hem = os.listdir(dirr_hem)

    for j in listdir_all:

        image_list.append(dirr_all + '/' + j)

        label.append(1)

    for j in listdir_hem:

        image_list.append(dirr_hem + '/' + j)

        label.append(0)

df = pd.DataFrame(list(zip(image_list, label)), columns=['Img', 'label'])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)



for train_index, test_index in split.split(df, df["label"]):

    strat_train = df.loc[train_index]

    strat_test = df.loc[test_index]



xTrain = np.empty(shape=(len(strat_train), fftn, fftm, 3))

xTest = np.empty(shape=(len(strat_test), fftn, fftm, 3))

yTrain = np.empty(shape=(len(strat_train), 2))

yTest = np.empty(shape=(len(strat_test), 2))



train_count = 0

test_count = 0



print('Тренировочные:')

for i in list(strat_train.index):

    # print(strat_train['Img'][i])

    img = preproc_resize(io.imread(strat_train['Img'][i]))

    xTrain[train_count, ] = img

    if strat_train['label'][i] == 1:

        yTrain[train_count, ] = [0, 1]

    if strat_train['label'][i] == 0:

        yTrain[train_count, ] = [1, 0]

    train_count += 1



plt.title('Кол-во больных на тренировочных данных')

plt.ylabel('кол-во')

strat_train['label'].value_counts().plot(kind='bar')

plt.show()



print('')

print('Тестовые:')

for i in list(strat_test.index):

    # print(strat_test['Img'][i])

    img = preproc_resize(io.imread(strat_test['Img'][i]))

    xTest[test_count,] = img

    if strat_test['label'][i] == 1:

        yTest[test_count,] = [0, 1]

    if strat_test['label'][i] == 0:

        yTest[test_count,] = [1, 0]

    test_count += 1





plt.title('Кол-во больных на тестовых данных')

plt.ylabel('кол-во')

strat_test['label'].value_counts().plot(kind='bar')

plt.show()
# модель нейронной сети

kernel_initializer = 'lecun_uniform'

bias_initializer = 'lecun_uniform'

kernel_regularizer = None

activation = "selu"



batch_size = 8

data_rows = fftn

data_cols = fftm

input_shape = (data_rows, data_cols, 3)

nb_epoch = 10  # learning epochs

alpha_zero = 0.001  # learning rate



model = models.Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape, border_mode="same", data_format="channels_last", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

model.add(Activation(activation))

model.add(MaxPooling2D(pool_size=(3, 3)))



model.add(Conv2D(64, (3, 3), input_shape=input_shape, border_mode="same", data_format="channels_last", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

model.add(Activation(activation))

model.add(MaxPooling2D(pool_size=(3, 3)))



model.add(Conv2D(128, (3, 3), input_shape=input_shape, border_mode="same", data_format="channels_last", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

model.add(Activation(activation))

model.add(MaxPooling2D(pool_size=(3, 3)))



model.add(Conv2D(256, (3, 3), input_shape=input_shape, border_mode="same", data_format="channels_last", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

model.add(Activation(activation))

model.add(Dropout(0.8))

model.add(MaxPooling2D(pool_size=(3, 3)))



# adding fully connected layers

model.add(Flatten())

model.add(Dense(256, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

model.add(Activation(activation))

model.add(Dropout(0.8))

model.add(Dense(128, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

model.add(Activation(activation))

model.add(Dropout(0.8))

model.add(Dense(64, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

model.add(Activation(activation))

model.add(Dropout(0.6))

model.add(Dense(32, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))



model.add(Dense(2))

model.add(Activation('sigmoid'))

model.summary()



optimizer1 = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)

model.compile(loss=["binary_crossentropy"], optimizer=optimizer1, metrics=["accuracy"])

history = model.fit(xTrain, yTrain, epochs=nb_epoch, verbose=2, validation_data=(xTest, yTest))



score = model.evaluate(xTest, yTest, verbose=0)

print("Точность: %f" % score[1])

yPred = model.predict(xTest)

CM = multilabel_confusion_matrix(yTest, np.round_(yPred))

print("Матрица неточностей ", CM)

print(model.summary())



def visualise(history):



    # summarize history for accuracy

    plt.plot(history.history['accuracy'], 'k-')

    plt.plot(history.history['val_accuracy'], 'k:')

    plt.title('Model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epochs')

    plt.legend(['accuracy', 'val_accuracy'], loc='center right')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'], 'k-')

    plt.plot(history.history['val_loss'], 'k:')

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epochs')

    plt.legend(['loss', 'val_loss'], loc='center right')

    plt.show()

    

    # summarize history for loss

    plt.plot(history.history['loss'], 'k')

    plt.plot(history.history['accuracy'], 'k:')

    plt.title('Model loss and accuracy')

    plt.ylabel('loss/accuracy')

    plt.xlabel('epochs')

    plt.legend(['loss', 'accuracy'], loc='center right')

    plt.show()



# predict class for new image



def pred_image(model, image):

    img = io.imread(image)

    new_img = resize(img, (fftn, fftm, 3), anti_aliasing=True)

    x = np.empty(shape=(1, fftn, fftm, 3))

    x[0,] = new_img

    yPred = model.predict(x)

    return print('yPred', yPred)



visualise(history)