import os

import numpy as np

from os import listdir

from skimage.transform import resize

from PIL import Image

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import keras

from tensorflow.keras.models import Sequential

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, StandardScaler

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D, BatchNormalization

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from sklearn.model_selection import KFold

from sklearn.metrics import classification_report

from tensorflow.keras.metrics import Precision, Recall



# Settings:

img_size = 64

grayscale_images = False

num_class = 10

test_size = 0.2

validation_size = 0.2



def get_img(data_path):

    # Getting image array from path:

    img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE if grayscale_images else cv2.IMREAD_COLOR)

    img = np.array(Image.fromarray(img).resize((img_size, img_size)))

    return img



def get_dataset(dataset_path='../input/dataset/Dataset'):

    try:

        X = np.load('npy_dataset/X.npy')

        Y = np.load('npy_dataset/Y.npy')

    except:

        labels = listdir(dataset_path) # Geting labels

        X = []

        Y = []

        for i, label in enumerate(labels):

            datas_path = dataset_path+'/'+label

            for data in listdir(datas_path):

                img = get_img(datas_path+'/'+data)

                X.append(img)

                Y.append(i)

        if grayscale_images:

            X = 1-np.array(X).astype('float32')/255.

        else:        

            imagesAvg = np.average(X)

            X = (X - imagesAvg)/255.

        Y = np.array(Y).astype('float32')

        Y = to_categorical(Y, num_class)

        if not os.path.exists('npy_dataset/'):

            os.makedirs('npy_dataset/')

        np.save('npy_dataset/X.npy', X)

        np.save('npy_dataset/Y.npy', Y)

    X, X_rest, Y, Y_rest = train_test_split(X, Y, test_size=test_size+validation_size, random_state=42)

    validationRatio = validation_size / (test_size + validation_size)

    X_test, X_validation, Y_test, Y_validation = train_test_split(X_rest, Y_rest, test_size=validationRatio, random_state=42)

    return X, X_test, X_validation, Y, Y_test, Y_validation



x_train, x_test, x_validation, y_train, y_test, y_validation = get_dataset()



x_test = x_test.reshape(x_test.shape[0],img_size,img_size, 1 if grayscale_images else 3)

x_train = x_train.reshape(x_train.shape[0],img_size,img_size, 1 if grayscale_images else 3)

x_validation = x_validation.reshape(x_validation.shape[0],img_size,img_size, 1 if grayscale_images else 3)


plt.subplot(1, 4, 1)

if(grayscale_images):

    plt.imshow(x_train[1].reshape(img_size, img_size))

else:

    plt.imshow(x_train[1].reshape(img_size, img_size, 3))

plt.axis('off')

plt.subplot(1, 4, 2)

if(grayscale_images):

    plt.imshow(x_train[2].reshape(img_size, img_size))

else:

    plt.imshow(x_train[2].reshape(img_size, img_size, 3))

plt.axis('off')
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def evaluate_model(model, x_train, x_test, x_validation, y_train, y_test, y_validation):

    model = model()

    model.fit(x_test, y_test, batch_size=32, epochs=3, validation_data=(x_validation, y_validation))

    print('Model evaluation ',model.evaluate(x_test,y_test))

    final_loss, final_acc, *t = model.evaluate(x_test, y_test, verbose = 0)

    print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

def gmodel1():

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())



    model.add(Dense(64))



    model.add(Dense(10))

    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                   metrics=[Precision(), Recall()])

    return model



evaluate_model(gmodel1, x_train, x_test, x_validation, y_train, y_test, y_validation)
def gmodel2():

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(Dense(128))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())



    model.add(Dense(64))



    model.add(Dense(10))

    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(gmodel2, x_train, x_test, x_validation, y_train, y_test, y_validation)
def gmodel3():

    model = Sequential()

    model.add(Dense(512))

    model.add(Conv2D(256, (3, 3), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(Dense(10))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(64))

    model.add(Flatten())



    



    model.add(Dense(10))

    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(gmodel3, x_train, x_test, x_validation, y_train, y_test, y_validation)
def gmodel4():

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(10))



    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(gmodel4, x_train, x_test, x_validation, y_train, y_test, y_validation)
def rgbmodel1():

    model = Sequential()

    model.add(Conv2D(256, (5, 5), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(10))



    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(rgbmodel1, x_train, x_test, x_validation, y_train, y_test, y_validation)
def rgbmodel2():

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(BatchNormalization())

    model.add(Conv2D(256, (1, 1), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (5, 5), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(10))



    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(rgbmodel2, x_train, x_test, x_validation, y_train, y_test, y_validation)
def rgbmodel3():

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv2D(256, (1, 1), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))





    model.add(Conv2D(256, (2, 2), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(10))



    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(rgbmodel3, x_train, x_test, x_validation, y_train, y_test, y_validation)
def rgbmodel4():

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv2D(256, (1, 1), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(BatchNormalization())

    model.add(Conv2D(256, (2, 2), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (10, 10), input_shape=x_train[0].shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(10))



    model.add(Activation('softmax'))



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=[Precision(), Recall()])

    return model



evaluate_model(rgbmodel4, x_train, x_test, x_validation, y_train, y_test, y_validation)