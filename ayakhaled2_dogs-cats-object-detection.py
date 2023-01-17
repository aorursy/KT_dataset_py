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
import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

from tqdm import tqdm



DATADIR = "/kaggle/input/dogscats/PetImages"



CATEGORIES = ["Dog", "Cat"]



for category in CATEGORIES:  # do dogs and cats

    path = os.path.join(DATADIR,category)  # create path to dogs and cats

    for img in os.listdir(path):  # iterate over each image per dogs and cats

        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array

        plt.imshow(img_array, cmap='gray')  # graph it

        plt.show()  # display!



        break  # we just want one for now so break

    break  #...and one more!
print(img_array)
print(img_array.shape)
IMG_SIZE = 50



new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

plt.imshow(new_array, cmap='gray')

plt.show()
training_data = []



def create_training_data():

    for category in CATEGORIES:  # do dogs and cats



        path = os.path.join(DATADIR,category)  # create path to dogs and cats

        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat



        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats

            try:

                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

                training_data.append([new_array, class_num])  # add this to our training_data

            except Exception as e:  # in the interest in keeping the output clean...

                pass

            #except OSError as e:

            #    print("OSErrroBad img most likely", e, os.path.join(path,img))

            #except Exception as e:

            #    print("general exception", e, os.path.join(path,img))



create_training_data()



print(len(training_data))
import random



random.shuffle(training_data)

for sample in training_data[:10]:

    print(sample[1])
X = []

y = []



for features,label in training_data:

    X.append(features)

    y.append(label)





X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#X = np.array(X).reshape(-1, IMG_SIZE)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

y = np.array(y)

import pickle



pickle_out = open("X.pickle","wb")

pickle.dump(X, pickle_out)

pickle_out.close()



pickle_out = open("y.pickle","wb")

pickle.dump(y, pickle_out)

pickle_out.close()
pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D



import pickle



pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = X/255.0



model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(64))

model.add(Dense(256, activation='relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from tensorflow.keras.callbacks import TensorBoard

import pickle

import time



NAME = "Cats-vs-dogs-CNN"



pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = X/255.0



model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))



model.add(Dense(1))

model.add(Activation('sigmoid'))



tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'],

              )



model.fit(X, y,

          batch_size=32,

          epochs=3,

          validation_split=0.3,

          callbacks=[tensorboard])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from tensorflow.keras.callbacks import TensorBoard

import pickle

import time



NAME = "Cats-vs-dogs-64x2-CNN"



pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = X/255.0



model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Activation('relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))



tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'],

              )



model.fit(X, y,

          batch_size=32,

          epochs=10,

          validation_split=0.3,

          callbacks=[tensorboard])

import time



dense_layers = [0,1,2]

layer_sizes = [32, 64, 128]

conv_layers = [1, 2, 3]



for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)

dense_layers = [0, 1, 2]

layer_sizes = [32, 64, 128]

conv_layers = [1, 2, 3]



for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)



            model = Sequential()



            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



            model.add(Flatten())

            for _ in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation('relu'))



            model.add(Dense(1))

            model.add(Activation('sigmoid'))
"""from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from tensorflow.keras.callbacks import TensorBoard

import pickle

import time



pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = X/255.0



dense_layers = [0, 1, 2]

layer_sizes = [32, 64, 128]

conv_layers = [1, 2, 3]



for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)



            model = Sequential()



            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



            model.add(Flatten())



            for _ in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation('relu'))



            model.add(Dense(1))

            model.add(Activation('sigmoid'))



            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



            model.compile(loss='binary_crossentropy',

                          optimizer='adam',

                          metrics=['accuracy'],

                          )



            model.fit(X, y,

                      batch_size=32,

                      epochs=10,

                      validation_split=0.3,

                      callbacks=[tensorboard])"""
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

from tensorflow.keras.callbacks import TensorBoard

import pickle

import time



pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = X/255.0



dense_layers = [0]

layer_sizes = [64]

conv_layers = [3]



for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)



            model = Sequential()



            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



            model.add(Flatten())



            for _ in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation('relu'))



            model.add(Dense(1))

            model.add(Activation('sigmoid'))



            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



            model.compile(loss='binary_crossentropy',

                          optimizer='adam',

                          metrics=['accuracy'],

                          )



            model.fit(X, y,

                      batch_size=32,

                      epochs=10,

                      validation_split=0.3,

                      callbacks=[tensorboard])



model.save('64x3-CNN.model')
import tensorflow as tf



CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value





def prepare(filepath):

    IMG_SIZE = 50  # 50 in txt-based

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('/kaggle/input/doggandcatt/dog.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

prediction

prediction[0][0]

print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('/kaggle/input/dogscats/PetImages/Cat/8097.jpg')])

print(prediction)  # will be a list in a list.

print(CATEGORIES[int(prediction[0][0])])