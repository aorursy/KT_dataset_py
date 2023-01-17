import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rn

import tensorflow as tf

import keras

import warnings

import numpy as np



from scipy.io import loadmat

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# %config IPCompleter.greedy = True
# warnings.filterwarnings('ignore')



# tf.set_random_seed(30)
mnist = loadmat('../input/mnist-original/mnist-original.mat')

mnist
x = mnist['data'].T

y = mnist['label'].T #[0]

print(x.shape)

print(y.shape)    

print(np.unique(y))   #we will do 1 hot encoding of these
import matplotlib.pyplot as plt

%matplotlib inline



idx = rn.randint(0, x.shape[0])

plt.imshow(x[idx].reshape(28,28).reshape(28,28), "gray") 
img_height = 28

img_width = 28

channels = 1



input_shape = (img_height, img_width, channels)

num_classes = 10



epoch = 20

batch_size = 128
x_reshape = x.reshape(x.shape[0], img_height, img_width, channels)



print(x_reshape.shape)
y_encoded = keras.utils.to_categorical(y, num_classes)



print(y_encoded.shape)
idx = rn.sample(range(0, len(y_encoded)), 10)

y_random = []

for i in idx:

    y_random.append([int(x) for x in y_encoded[i]])



y_random
x_reshape = x_reshape.astype('float32')

x_reshape /= 255
x_train, x_test, y_train, y_test = train_test_split(x_reshape, y_encoded, test_size = 0.25, random_state = 0)
print('training data shape : image - {0}, label - {1}'.format(x_train.shape, y_train.shape))

print('test data shape : image - {0}, label - {1}'.format(x_test.shape, y_test.shape))
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)
print('training data shape : image - {0}, label - {1}'.format(x_train.shape, y_train.shape))

print('validation data shape : image - {0}, label - {1}'.format(x_validation.shape, y_validation.shape))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
# model

model = Sequential()



# first conv layer

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))



# second conv layer

model.add(Conv2D(64, kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# flatten and put a fully connected layer

model.add(Flatten())

model.add(Dense(128, activation='relu')) # fully connected

model.add(Dropout(0.5))



# softmax layer

model.add(Dense(num_classes, activation='softmax'))



# model summary

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
history = model.fit(x_train, 

         y_train,

         batch_size = batch_size,

         epochs = epoch,

         verbose = 1,

         validation_data=(x_validation, y_validation))
def plot_history(history):   # Plot training & validation accuracy values

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



plot_history(history)
ev=model.evaluate(x_test, y_test)

print(ev)

print(model.metrics_names)
def create_model(additioanl_conv_layers=1, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), epoch=10, skip_dropout_conv=False, skip_maxpool=False):

    # model

    model = Sequential()



    # first conv layer

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))



    for i in range(additioanl_conv_layers):

        # additional conv layers

        model.add(Conv2D(64, kernel_size=(3, 3), 

                         activation='relu'))

        if not skip_maxpool:

            model.add(MaxPooling2D(pool_size=(2, 2)))

        if not skip_dropout_conv:

            model.add(Dropout(0.25))



    # flatten and put a fully connected layer

    model.add(Flatten())

    model.add(Dense(128, activation='relu')) # fully connected

    model.add(Dropout(0.5))



    # softmax layer

    model.add(Dense(num_classes, activation='softmax'))



    # model summary

    print(model.summary())



    model.compile(loss=loss,

                  optimizer=optimizer,

                  metrics=['accuracy'])

    history = model.fit(x_train, 

         y_train,

         batch_size = batch_size,

         epochs = epoch,

         verbose = 1,

         validation_data=(x_validation, y_validation))

    ev = model.evaluate(x_test, y_test)

    print(ev)

    print(model.metrics_names)

    plot_history(history)

    return ev
results={}

results["0"]=create_model(additioanl_conv_layers=0, optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True), skip_dropout_conv=False, skip_maxpool=False)

results["1"]=create_model(additioanl_conv_layers=1, optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True), skip_dropout_conv=False, skip_maxpool=False)

results["2"]=create_model(additioanl_conv_layers=2, optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True), skip_dropout_conv=False, skip_maxpool=False)

results["3"]=create_model(additioanl_conv_layers=3, optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True), skip_dropout_conv=False, skip_maxpool=False)

print(results)