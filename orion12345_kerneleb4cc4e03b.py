import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

import numpy as np

import h5py

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

import pandas as pd

import h5py

import numpy as np

import tensorflow as tf

import math

import numpy as np

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

data=pd.read_csv('../input/train.csv').as_matrix()

y=data[0:,0]

data=data[0:,1:]







# Standardize data to have feature values between 0 and 1.

X_train = (data[0:37000,0:])/255.

X_test=(data[37000:,0:])/255.

X_train=np.reshape(X_train,(37000,28,28))



X_test=np.reshape(X_test,(5000,28,28))
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y

Y_train = convert_to_one_hot(y[0:37000], 10)

Y_train=Y_train.T

Y=Y_test

Y_test = convert_to_one_hot(y[37000:], 10)

Y_test=Y_test.T
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import Sequential

model=Sequential()

model.add(Conv2D(64,(3,3),strides=1,name='c0',input_shape=(28,28,1)))

model.add(BatchNormalization(axis=3,name='b0'))

model.add(Activation('relu'))



model.add(Conv2D(64,(3,3),strides=1,name='c1'))

model.add(BatchNormalization(axis=3,name='b1'))

model.add(Activation('relu'))



model.add(AveragePooling2D((2,2),name='m0'))







model.add(Conv2D(128,(3,3),strides=1,name='c2'))

model.add(BatchNormalization(axis=3,name='b2'))

model.add(Activation('relu'))



model.add(Conv2D(128,(3,3),strides=1,name='c3'))

model.add(BatchNormalization(axis=3,name='b3'))

model.add(Activation('relu'))



model.add(AveragePooling2D((2,2),name='m1'))



model.add(Conv2D(256,(3,3),strides=1,name='c4'))

model.add(BatchNormalization(axis=3,name='b4'))

model.add(Activation('relu'))



model.add(AveragePooling2D((2,2),name='m2'))



model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(512,activation="relu"))

    

model.add(Dense(10,activation="softmax"))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )



# Compiling the model

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])













datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
train_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)

test_gen = datagen.flow(X_test, Y_test, batch_size=batch_size)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2],1))

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
history = model.fit_generator(train_gen, 

                              epochs = epochs, 

                              steps_per_epoch = X_train.shape[0] // batch_size,

                              validation_data = test_gen,

                              validation_steps = X_test.shape[0] // batch_size)
preds = model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

### END CODE HERE ###

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
data1=pd.read_csv('../input/test.csv').as_matrix()



data1=(data1)/255.

data1=np.reshape(data1,(28000,28,28))
data1=np.reshape(data1,(data1.shape[0],data1.shape[1],data1.shape[2],1))
pre = model.predict_classes(data1, verbose=1)

print(pre)
import os

print(os.listdir('../input'))


pre= pd.Series(pre,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pre],axis = 1)



submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)