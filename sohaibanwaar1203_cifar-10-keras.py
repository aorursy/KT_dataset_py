# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers import  BatchNormalization

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold

from six.moves import cPickle as pickle

from keras import regularizers

import os

import platform

from subprocess import check_output

import matplotlib.pyplot as plt

import argparse

import random

import cv2

import os

from keras.layers.core import Dense, Dropout, Flatten



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#making class of lenet neural network

class leNet:

    def initilize_model(height, width, depth,classes):

        model=Sequential()

        input_shape=(height,width,depth)

        # if we are using "channel first" than our first arugemnt of input_shape will change to depth

        if K.image_data_format()=="channels first":

            input_shape=(depth,height,width)

        # conv2d set  =====> Conv2d====>relu=====>MaxPooling

        model.add(Conv2D(20,(5,5),padding="same"))

        model.add(Activation("relu"))

        #model.add(BatchNormalization())

        #model.add(Conv2D(20,(3,3),padding="same"))

        #model.add(Activation("relu"))

        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Dropout(0.2))

        

        # conv2d set2  =====> Conv2d====>relu=====>MaxPooling

        model.add(Conv2D(50,(5,5),padding="same"))

        model.add(Activation("relu"))

        #model.add(BatchNormalization())

        #model.add(Conv2D(50,(3,3),padding="same"))

        #model.add(Activation("relu"))

        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Dropout(0.5))

        

        model.add(Conv2D(80,(5,5),padding='same'))

        model.add(Activation("relu"))

        #model.add(BatchNormalization())

        #model.add(Conv2D(80,(5,5),padding='same'))

        #model.add(Activation("relu"))

        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Dropout(0.2))

        

        model.add(Conv2D(110,(5,5),padding='same'))

        model.add(Activation("relu"))

        #model.add(BatchNormalization())

        #model.add(Conv2D(110,(5,5),padding='same'))

        #model.add(Activation("relu"))

        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Dropout(0.6))

        

        #now adding fully connected layer

        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation("relu"))

        model.add(Dense(512))

        model.add(Activation("relu"))

        

        #now adding Softmax Classifer because we want to classify 10 class

        

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        

        return model
def load_pickle(f):

    version = platform.python_version_tuple()

    if version[0] == '2':

        return  pickle.load(f)

    elif version[0] == '3':

        return  pickle.load(f, encoding='latin1')

    raise ValueError("invalid python version: {}".format(version))



def load_CIFAR_batch(filename):

    """ load single batch of cifar """

    with open(filename, 'rb') as f:

        datadict = load_pickle(f)

        X = datadict['data']

        Y = datadict['labels']

        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

        Y = np.array(Y)

        return X, Y



def load_CIFAR10(ROOT):

    """ load all of cifar """

    xs = []

    ys = []

    for b in range(1,6):

        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))

        X, Y = load_CIFAR_batch(f)

        xs.append(X)

        ys.append(Y)

    Xtr = np.concatenate(xs)

    Ytr = np.concatenate(ys)

    del X, Y

    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):

    # Load the raw CIFAR-10 data

    cifar10_dir = '../input/cifar10'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)



    # Subsample the data

    mask = range(num_training, num_training + num_validation)

    X_val = X_train[mask]

    y_val = y_train[mask]

    mask = range(num_training)

    X_train = X_train[mask]

    y_train = y_train[mask]

    mask = range(num_test)

    X_test = X_test[mask]

    y_test = y_test[mask]

    

    # Normalize the data: subtract the mean image

    #Zscores

    std = np.std(X_train,axis=(0,1,2,3))

    std = np.std(X_val,axis=(0,1,2,3))

    std = np.std(X_test,axis=(0,1,2,3))

    mean_image = np.mean(X_train, axis=0)

    X_train =(X_train - mean_image)/(std)

    X_val = (X_val - mean_image)/std

    X_test = (X_test - mean_image)/std



    return X_train, y_train, X_val, y_val, X_test, y_test





# Invoke the above function to get our data.

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

print('Train data shape: ', X_train.shape)

print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_val.shape)

print('Validation labels shape: ', y_val.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)

y=y_train

y_train = to_categorical(y_train, num_classes=10)

y_test = to_categorical( y_test, num_classes=10)

y_val = to_categorical( y_val, num_classes=10)
EPOCHS = 5

INIT_LR = 1e-3

BS = 32

#data augmentation

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    )

datagen.fit(X_train)



# Model

model = leNet.initilize_model(width=32, height=32, depth=3, classes=10)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

   # Compile the model

model.compile(loss='categorical_crossentropy',

                  optimizer=Adam(lr=0.0001, decay=1e-6),

                  metrics=['accuracy'])
history= model.fit(X_train, y_train,

              batch_size=128,

              shuffle=True,

              epochs=10,

              validation_data=(X_val, y_val),

              

             )
scores = model.evaluate(X_test,y_test)



print('Loss: %.3f' % scores[0])

print('Accuracy: %.3f' % scores[1])
print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.summary()
def predict_label(a): # a is the number of index for which model predict the max value

    if a==0:p="Aeroplane"

    if a==1: p="automobile"

    if a==2: p="bird"

    if a==3: p="cat"

    if a==4: p="deer"

    if a==5: p="dog"

    if a==6: p="frog"

    if a==7: p="hourse"

    if a==8: p="Ship"

    if a==9: p="truck"

    

    return p
x = np.expand_dims(X_test[45], axis=0)

a=model.predict(x)

maxindex = a.argmax()

p=predict_label(maxindex)

plt.title("Model Prediction ="+p)

plt.imshow(X_test[45])

from sklearn.metrics import classification_report

import numpy as np



Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index

y_pred = model.predict_classes(X_test)

print(classification_report(Y_test, y_pred))