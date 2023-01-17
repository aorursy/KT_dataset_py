import cv2 as cv2 

import os

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD

from keras import backend as K

K.set_image_data_format('channels_first')

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import tensorflow as tf 



np.random.seed(1)

finalAct = 'softmax' 

IMG_SIZE = 48

NUM_CLASSES = int(8)
def init_img(img):

    #histogram normalization

    res = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    res[:, :, 0] = cv2.equalizeHist(res[:, :, 0])

    res = cv2.cvtColor(res, cv2.COLOR_YUV2BGR)



    #resize image

    res = cv2.resize(res, (IMG_SIZE, IMG_SIZE))



    #roll axis

    res = np.rollaxis(res, -1)



    return res
def load_image_from_folder(folder, arr):

    len = 0

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename), 1)

        if img is not None:

            arr.append(init_img(img))

            len+=1

    return arr, len



X = []

Y = []

labels = []



for i in range(8):

    X, num = load_image_from_folder('train/' + str(i), X)

    for j in range(num):

        labels.append(i)



X = np.array(X, dtype = 'float32')

per = np.random.permutation(X.shape[0])



tmp_X = []

tmp_Y = []



i = 0 

for j in per:

        tmp_X.append(X[j])

        tmp_Y.append(labels[j])



X = tmp_X

labels = tmp_Y



X = np.array(X, dtype = 'float32')

labels = np.array(labels, dtype = 'uint8')

Y = np.eye(NUM_CLASSES, dtype = 'uint8')[labels]

X = tf.keras.utils.normalize(X, axis=1)



def cnn_model():

    model = Sequential() #Init

    #Convolutinal Layer, we'll use a 3x3 kernel to load through our image, and return feature map

    model.add(Conv2D(32, (3, 3), padding='same',

                     input_shape=(3, IMG_SIZE, IMG_SIZE),

                     activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    #MaxPolling Layer, it help us to reduce spatial dimension

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), padding='same',

                     activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, (3, 3), padding='same',

                     activation='relu'))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    #Flat to prepare for the next fully connection layer

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    #finalAct = softmax, it will return a tuple, in this tuple that n-class size, i-th element mean the probality to become ith-class

    #and sum of them = 1

    model.add(Dense(NUM_CLASSES, activation=finalAct))

    return model



model = cnn_model()
lr = 0.01

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])

batch_size = 32

epochs = 30



model_trained = model.fit(X, Y,

          batch_size=batch_size,

          epochs=epochs,

          validation_split=0.2,

          callbacks=[LearningRateScheduler(lr_schedule),

                     ModelCheckpoint('model.h5', save_best_only=True)]

          )

          

model.save('my_model.h5')
from keras.models import load_model

import cv2

import numpy as np 

import tensorflow as tf 

import os



#Load our model that have been fitted before

model = load_model('model.h5')

PATH = 'data_private'



IMG_SIZE = 48



#reuse our init function in the training phase

def init_img(img):

    #histogram normalization

    res = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    res[:, :, 0] = cv2.equalizeHist(res[:, :, 0])

    res = cv2.cvtColor(res, cv2.COLOR_YUV2BGR)



    #resize image

    res = cv2.resize(res, (IMG_SIZE, IMG_SIZE))



    #roll axis

    res = np.rollaxis(res, -1)



    return res



#yeah, reused it again

def load_image_from_folder(folder):

    name = []

    arr = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename), 1)

        if img is not None:

            arr.append(init_img(img))

            name.append(filename)

    return arr, name



img, filename = load_image_from_folder(PATH)



#turn list data to ndarray, and normalize them

test_data = np.array(img, dtype = 'float32')

test_data = tf.keras.utils.normalize(test_data, axis=1)



#1 line to predict our test data, return a ndarray of labels

result = model.predict_classes(test_data)



#write to "data_output.csv" file

fout = open('data_output.csv', 'w')

for i in range(len(test_data)):

    fout.write(filename[i] + ',' + str(result[i]) + '\n')