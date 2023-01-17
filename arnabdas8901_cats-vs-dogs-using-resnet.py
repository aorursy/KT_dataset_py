#Imports

import keras

from keras_preprocessing.image import ImageDataGenerator

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

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)
#Unzip zipped data set

from zipfile import ZipFile

with ZipFile('/kaggle/input/dogs-vs-cats/train.zip','r') as zip:

    zip.extractall('/kaggle/working')

with ZipFile('/kaggle/input/dogs-vs-cats/test1.zip','r') as zip:

    zip.extractall('/kaggle/working')
#Storing Cats and dogs training images in respective folder.

%mkdir /kaggle/working/train/Cat

%mv /kaggle/working/train/cat* /kaggle/working/train/Cat/

%mkdir /kaggle/working/train/Dog

%mv /kaggle/working/train/dog* /kaggle/working/train/Dog/

%mkdir /kaggle/working/test1/test

%mv -f /kaggle/working/test1/*.jpg /kaggle/working/test1/test/
taining_data = '/kaggle/working/train'

test_data = '/kaggle/working/test1'
training_batch = ImageDataGenerator(rescale=1./255).flow_from_directory(taining_data, target_size=(256, 256), color_mode='rgb', class_mode='categorical', batch_size=32)

test_batch = ImageDataGenerator(rescale=1./255).flow_from_directory(test_data, target_size=(256, 256), color_mode='rgb', class_mode=None, batch_size=32)
def identity_block(X, f, filters):

    F1, F2, F3 = filters

    

    X_skip = X

    

    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    

    

    # Second component of main path 

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X,X_skip])

    X = Activation('relu')(X)

    

    ### END CODE HERE ###

    

    return X
def convolutional_block(X, f, filters, s = 2):

    

    F1, F2, F3 = filters

    

    # Save the input value

    X_skip = X



    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    



    # Second component of main path

    X = Conv2D(F2, (f, f), strides = (1,1), padding='same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(F3, (1, 1), strides = (1,1), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)



    X_skip = Conv2D(F3, (1, 1), strides = (s,s), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_skip)

    X_skip = BatchNormalization(axis = 3)(X_skip)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation 

    X = Add()([X,X_skip])

    X = Activation('relu')(X)

    

    return X
def ResNetModel(input_shape):



    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)

    X = identity_block(X, 3, [64, 64, 256])

    X = identity_block(X, 3, [64, 64, 256])



    # Stage 3

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)

    X = identity_block(X, 3, [128, 128, 512])

    X = identity_block(X, 3, [128, 128, 512])

    X = identity_block(X, 3, [128, 128, 512])



    # AVGPOOL

    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    #MAXPool

    

    X = MaxPooling2D((2, 2), strides=(2, 2))(X)



    # output layer

    X = Flatten()(X)

    X = Dense(128, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

    X = Dense(128, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

    X = Dense(2, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)

    

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNetModel')



    return model
myClassifier = ResNetModel((256, 256, 3))
myClassifier.summary()
myClassifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
myClassifier.fit(training_batch, epochs = 40, verbose=2)
myClassifier.save('/kaggle/working/CatsVsDogs_resnet_Custom.h5')

myClassifier.save_weights('/kaggle/working/CatsVsDogs_resnet_Custom_weghts.h5')
import numpy as np

import os

import csv



directory = '/kaggle/working/test1/test'



with open('mySubmision.csv', 'w', newline='') as op:

    myWritter = csv.writer(op)

    myWritter.writerow(['id','label'])

    for filename in os.listdir(directory):

        img = image.load_img(os.path.join(directory,filename), target_size=(256, 256))

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        x = x/255.0

        id = filename.split('.')[0]

        prediction = np.argmax(myClassifier.predict(x))

        myWritter.writerow([id,prediction])