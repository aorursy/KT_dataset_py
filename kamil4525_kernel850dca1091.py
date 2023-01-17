import numpy as np

import pandas as pd 

import os

import cv2

import matplotlib.pyplot as plt

import random

import pickle



from keras.utils import np_utils
train_dir = "../input/oct2017/OCT2017 /train"

validation_dir = "../input/oct2017/OCT2017 /val"

test_dir = "../input/oct2017/OCT2017 /test"



CATEGORIES = ["DME","CNV","NORMAL","DRUSEN"]

IMG_SIZE = 50
# from TRAIN data

for category in CATEGORIES:

    path = os.path.join(train_dir, category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array, cmap="gray")

        plt.show()

        break

    break
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

plt.imshow(new_array, cmap="gray")

plt.show()
print(new_array)
training_data = []

validation_data = []

test_data = []



def get_data(dataset_path):

    data_array = []

    for category in CATEGORIES:

        path = os.path.join(dataset_path, category)

        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                data_array.append([new_array, class_num])

            except Exception as e:

                pass

    return data_array



training_data = get_data(train_dir)

validation_data = get_data(validation_dir)

test_data = get_data(test_dir)
print("training data length: ", len(training_data))

print("validation data length: ", len(validation_data))

print("testing data length: ", len(test_data))
# schuffle training data

random.shuffle(training_data)



for sample in training_data[:5]:

    print(sample[0])
x_train = []

y_train = []



x_test = []

y_test = []



# split data to x and y

for features, label in training_data:

    x_train.append(features)

    y_train.append(label)

    

for features, label in test_data:

    x_test.append(features)

    y_test.append(label)

    

print(x_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))



# all samples should have the same size

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out = open("x_train.pickle", "wb")

pickle.dump(x_train, pickle_out)

pickle_out.close()



pickle_out = open("y_train.pickle", "wb")

pickle.dump(y_train, pickle_out)

pickle_out.close()



pickle_out = open("x_test.pickle", "wb")

pickle.dump(x_test, pickle_out)

pickle_out.close()



pickle_out = open("y_test.pickle", "wb")

pickle.dump(y_test, pickle_out)

pickle_out.close()
def load_dataset():

    pickle_in = open("x_train.pickle", "rb")

    x_train = pickle.load(pickle_in)



    pickle_in = open("y_train.pickle", "rb")

    y_train = pickle.load(pickle_in)



    pickle_in = open("x_test.pickle", "rb")

    x_test = pickle.load(pickle_in)



    pickle_in = open("y_test.pickle", "rb")

    y_test = pickle.load(pickle_in)

    

    # one hot encoding

    y_train = np_utils.to_categorical(y_train)

    y_test = np_utils.to_categorical(y_test)

    

    return x_train, y_train, x_test, y_test



def scale_pixels(train, test):

    # convert: integers -> float32

    train_norm = train.astype('float32')

    test_norm = test.astype('float32')

    

    # normalize to range 0-1

    train_norm = train_norm / 255.0

    test_norm = test_norm / 255.0

    

    # return normalized images

    return train_norm, test_norm
def show_summary(history):

    # plot loss

    plt.subplot(211)

    plt.title('Cross Entropy Loss')

    plt.plot(history.history['loss'], color='blue', label='train')

    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy

    plt.subplot(212)

    plt.title('Classification Accuracy')

    plt.plot(history.history['acc'], color='blue', label='train')

    plt.plot(history.history['val_acc'], color='orange', label='test')

    

def run_test(mod, iterations = None):

    # load dataset

    trainX, trainY, testX, testY = load_dataset()

    

    # scale pixels

    trainX, testX = scale_pixels(trainX, testX)

    

    if iterations is None:

        iterations = 100

    

    # fit model

    history = mod.fit(trainX, trainY, 

                        epochs = iterations, 

                        batch_size = 64, 

                        validation_data = (testX, testY), 

                        verbose = 0)



    # evaluate model

    _, acc = mod.evaluate(testX, testY, verbose = 0)



    # print accuracy

    print('Accuracy (on testing set): > %.3f' % (acc * 100.0))

    

    # return history

    return history
from keras.datasets import cifar10

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD

from keras.layers import Dropout

from keras.layers import BatchNormalization



# define cnn model

def define_model_v3_dropout_normalization():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))

    # compile model

    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



model = define_model_v3_dropout_normalization()



# run test

history = run_test(model, 400)
show_summary(history)