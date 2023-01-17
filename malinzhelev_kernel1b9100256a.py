import os

dir = '/kaggle/input/general-methods/'

for dirname, _, filenames in os.walk(dir):

    for filename in filenames:

        with open(dir + filename, 'rb') as file_pi:

            data = pickle.load(file_pi)

            

            pyplot.plot(data['accuracy'])

            pyplot.xlabel("Epoch")

            pyplot.ylabel("Model accuracy")

            pyplot.savefig(filename + '_acc.png')

            pyplot.plot(data['val_accuracy'])

            pyplot.xlabel("Epoch")

            pyplot.ylabel("Model accuracy")

            pyplot.savefig(filename + '_test_acc.png')

            pyplot.plot(data['loss'])

            pyplot.xlabel("Epoch")

            pyplot.ylabel("Value of loss function")

            pyplot.savefig(filename + '_loss.png')

            pyplot.plot(data['val_loss'])

            pyplot.xlabel("Epoch")

            pyplot.ylabel("Value of loss function")

            pyplot.savefig(filename + '_test_loss.png')

print("Done")
from matplotlib import pyplot

with open('/kaggle/working/historyFinal', 'rb') as file_pi:

    data = pickle.load(file_pi)



pyplot.plot(data['accuracy'])

pyplot.xlabel("Epoch")

pyplot.ylabel("Model accuracy")

pyplot.show()

import numpy as np

import random

import tensorflow as tf

import numpy as np

import random

import pickle

from datetime import datetime

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Convolution2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot

import time



def run_regularized():

    history = run_model(

        lr = 0.01,

        opt = 'SGD',

        early_stop = True,

        conv_layers = 1,

        deep_layers = 1,

        regularize = True

    )



    with open('/kaggle/working/regularized' + filename, 'wb') as file_pi:

        pickle.dump(history.history, file_pi)



def run_general():

    one_conv_one_deep = run_final(c_l=1, d_l=1, filename='OneConvOneDeep')

    one_conv_two_deep = run_final(c_l=1, d_l=2, filename='OneConvTwoDeep')

    two_conv_one_deep = run_final(c_l=2, d_l=1, filename='TwoConvOneDeep')

    two_conv_two_deep = run_final(c_l=2, d_l=2, filename='TwoConvTwoDeep')

    

    print("Time elapsed for 1 convolutional and 1 deep layer:")

    print(one_conv_one_deep)

    print("Time elapsed for 1 convolutional and 2 deep layer:")

    print(one_conv_two_deep)

    print("Time elapsed for 2 convolutional and 1 deep layer:")

    print(two_conv_one_deep)

    print("Time elapsed for 2 convolutional and 2 deep layer:")

    print(two_conv_two_deep)

    

def run_final(c_l, d_l, filename):

    start_time = time.time()

    

    SGD_model = run_model(

        lr = 0.01,

        opt = 'SGD',

        early_stop = True,

        conv_layers = c_l,

        deep_layers = d_l

    )



    sgd_elapsed = time.time() - start_time



    with open('/kaggle/working/history' + filename, 'wb') as file_pi:

        pickle.dump(SGD_model.history, file_pi)

        

    return sgd_elapsed



def compare_optimizers():

    start_time = time.time()

    Adam_model = run_model(

        lr = 0.01,

        opt = 'adam'

    )

    adam_elapsed = time.time() - start_time

    

    with open('/kaggle/working/historyAdam', 'wb') as file_pi:

        pickle.dump(Adam_model.history, file_pi)



    start_time = time.time()

    

    SGD_model = run_model(

        lr = 0.01,

        opt = 'SGD'

    )



    sgd_elapsed = time.time() - start_time



    with open('/kaggle/working/historySGD', 'wb') as file_pi:

        pickle.dump(SGD_model.history, file_pi)

        

    print("SGD model execution time:")

    print(sgd_elapsed)

    print("Adam model execution time:")

    print(adam_elapsed)



def run_model(num_epochs=50, 

    data_augmenting=True, 

    lr=0.01, 

    conv_layers=1, 

    deep_layers=1,

    early_stop=False, 

    opt = 'adam'):



    # The path to the dataset

    data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/'

    

    # Data augmentation

    training_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        zca_whitening = True,

        fill_mode = 'reflect',

        rotation_range = 2,

        horizontal_flip=True,

        validation_split=0.25

    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = training_datagen.flow_from_directory(

        data_dir + 'train',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')

    test_set = test_datagen.flow_from_directory(

        data_dir + 'test',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



    # define model

    classifier = Sequential()





    # Add the convolutional layers (and pooling)

    for i in range(conv_layers):

        classifier.add(Convolution2D(32,(3,3),input_shape=(64, 64, 3), activation = 'relu'))

        classifier.add(MaxPooling2D((2,2),padding="same"))

        

    classifier.add(Flatten())



    # Add the deep layers (and the input one)

    input_layers = (64 * 64) / (conv_layers * 4)

    for i in range(deep_layers+1):

        classifier.add(Dense(input_layers, activation='relu'))



    classifier.add(Dense(1, activation='sigmoid'))



    #Set the optimizer

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    if(opt=='adam'):

        optimizer = keras.optimizers.Adam(learning_rate=lr)



    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    cb = []

    

    # Add early stopping

    if(early_stop):

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        cb.append(es)



    print(training_set)

    # Fit model

    model = classifier.fit(training_set, epochs=num_epochs, validation_data=test_set, callbacks=cb)

    

    return model

run_general()
compare_optimizers()