import os,cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pylab import rcParams

rcParams['figure.figsize'] = 10, 10



from sklearn.utils import shuffle

import keras



from keras.utils import np_utils



from keras import backend as K



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

from keras.applications import VGG19

#Load the VGG model

vgg_conv = VGG19(weights=None, include_top=False, input_shape=(64, 64,3))
def vgg_custom():

    model = Sequential()

    #add vgg conv model

    model.add(vgg_conv)

    

    #add new layers

    model.add(Flatten())

    model.add(Dense(1,  kernel_initializer='normal'))

    #model.compile(loss='mean_squared_error', optimizer=sgd())

    model.compile(loss='mean_squared_error', optimizer='adam')

    

    return model
classifier = vgg_custom()

classifier.summary()
#Data augmentation

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/training_set', 

                                                    target_size = (64, 64), 

                                                    batch_size = 32,

                                                   class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/test_set',

                                                target_size = (64, 64),

                                                 batch_size = 32, 

                                                 class_mode = 'binary')



hist = classifier.fit_generator(training_set, 

                         samples_per_epoch = 521, 

                        nb_epoch = 13, 

                        validation_data = test_set, 

                        nb_val_samples = 64)
# visualizing losses and accuracy

# %matplotlib inline



train_loss=hist.history['loss']

val_loss=hist.history['val_loss']

#train_acc=hist.history['acc']

#val_acc=hist.history['val_acc']



epochs = range(len(val_loss))



plt.plot(epochs,train_loss,'r-o', label='train_loss')

plt.plot(epochs,val_loss,'b', label='val_loss')

plt.title('train_loss vs val_loss')

#plt.plot(epochs,train_loss,'r-o', label='train_acc')

#plt.plot(epochs,val_loss,'b', label='val_acc')

#plt.title('train_acc vs val_acc')

plt.title('train_loss vs val_loss')

plt.legend()

plt.figure()

#plt.savefig('train_test_acc.png')

plt.savefig('train_test.png')