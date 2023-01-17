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
from numpy.random import seed

seed(1)

import tensorflow as tf

tf.compat.v1.set_random_seed

import random as rn

rn.seed(4)



import numpy as np

import keras

import sys

import pandas as pd

import matplotlib

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import to_categorical

import matplotlib.pyplot as plt

from keras.layers import BatchNormalization

from keras.constraints import unit_norm

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

import matplotlib.cm as cm



df_train = pd.read_csv("../input/digit-recognizer/train.csv")

#print(df_train.head())

df_test = pd.read_csv("../input/digit-recognizer/test.csv")



#(train_x, train_y), (test_x, test_y) = mnist.load_data(label_mode='fine')





dtr = df_train.values

dtest = df_test.values

train_x = dtr[:,1:]

train_y = dtr[:,0]

#val_x = dtr[9000:,:-1]

#val_y = dtr[9000:,-1]

test_x = dtest[:,:]

#test_y = dtest[:,-1]





#Preprocessing

#train_x = train_x.reshape((train_x.shape[0], 3, -1))

#test_x = test_x.reshape((test_x.shape[0], 3, -1))

#val_x = val_x.reshape((test_x.shape[0], 1, -1))



train_x = train_x/255.0

test_x = test_x/255.0

#val_x = val_x/255.0

#print(train_x.shape)

#train_x = np.concatenate([train_x[:, i, None].reshape(train_x.shape[0], -1, 1) for i in range(3)], axis=2)

#test_x = np.concatenate([test_x[:, i, None].reshape(test_x.shape[0], -1, 1) for i in range(3)], axis=2)

#val_x = np.concatenate([val_x[:, i, None].reshape(val_x.shape[0], -1, 1) for i in range(3)], axis=2)



train_x = train_x.reshape((train_x.shape[0],28,28,1))

test_x = test_x.reshape((test_x.shape[0],28,28,1))





print(train_x.shape,test_x.shape)





train_y = to_categorical(train_y)

print(train_y.shape)

#test_y = to_categorical(test_y)



#plt.imsave('filename.png', train_x[0].reshape(28,28), cmap=cm.gray)
def lrcheck(epoch):

    lr = 0.001

    if epoch > 75:

        lr = 0.0005

    if epoch > 100:

        lr = 0.0003

    return lr







weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(28,28,1)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

 

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))

 

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))

 

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

 

model.summary()

'''datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        zca_epsilon=1e-06,  # epsilon for ZCA whitening

        rotation_range= 0,  # randomly rotate images in the range (degrees, 0 to 180)

        # randomly shift images horizontally (fraction of total width)

        width_shift_range=0.1,

    # randomly shift images vertically (fraction of total height)

    height_shift_range=0.1,

    shear_range=0.,  # set range for random shear

    zoom_range=0.,  # set range for random zoom

    channel_shift_range=0.,  # set range for random channel shifts

    # set mode for filling points outside the input boundaries

    fill_mode='nearest',

    cval=0.,  # value used for fill_mode = "constant"

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False,  # randomly flip images

        # set rescaling factor (applied before any other transformation)

    rescale=None,

        # set function that will be applied on each input

    preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

    data_format=None,

        # fraction of images reserved for validation (strictly between 0 and 1)

    validation_split=0.0)



    # Compute quantities required for feature-wise normalization

    # (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(train_x)'''

    # Fit the model on the batches generated by datagen.flow().

''' history = model.fit_generator(datagen.flow(train_x, train_y,

                                     batch_size = 400),

                                     epochs = 50,

                                     workers=4,

                                     steps_per_epoch = 120)'''



batch_size = 400



opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

#history = model.fit_generator(datagen.flow(train_x, train_y, batch_size= batch_size),\

#                   steps_per_epoch = 1000, epochs = 40,\

#                    verbose = 1, callbacks = [LearningRateScheduler(lrcheck)])

history = model.fit(train_x,train_y, validation_split = 0.10, batch_size = batch_size, epochs = 3,verbose = 1, callbacks = [LearningRateScheduler(lrcheck)], shuffle=True)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.savefig('accuracy.png')

plt.show()





plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss') 

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.savefig('loss.png')

plt.show()



def convert(train_y):

    target = np.zeros((len(train_y),1), dtype = int)

    for i in range(len(train_y)):

        target[i][0] = int(np.argmax(train_y[i]))

    return target



prediction = model.predict(test_x)

target = convert(prediction)





np.savetxt("predictions.txt", target, delimiter = '\n')