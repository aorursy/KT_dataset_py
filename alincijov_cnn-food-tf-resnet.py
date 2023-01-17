import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



import os

import glob

import cv2
import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD
foods = list(os.walk('../input/food41/images/'))[0][1]

np.random.shuffle(foods)
# select how many types of foods to train on

nr_foods = 5
idx_to_name = {i:x for (i,x) in enumerate(foods[:nr_foods])}

name_to_idx = {x:i for (i,x) in enumerate(foods[:nr_foods])}



idx_to_name
data = []

labels = []

img_size = (112, 112)



for food in idx_to_name.values():

    path = '../input/food41/images/'

    imgs = [cv2.resize(cv2.imread(img), img_size, interpolation=cv2.INTER_AREA) for img in glob.glob(path + food + '/*.jpg')]

    for img in imgs:

        labels.append(name_to_idx[food])

        data.append(img)

        

# Normalize data

data = np.array(data)

data = data / 255.0

data = data.astype('float32')



# Create one hot encoding for labels

labels = np.array(labels)

labels = np.eye(len(idx_to_name.keys()))[list(labels)]
# check out the shapes

data.shape, labels.shape
# split training, labels

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
def bottleneck_residual_block(X, kernel_size, filters, reduce=False, s=2):

    # unpack the tuple to retrieve Filters of each CONV layer

    F1, F2, F3 = filters

    # Save the input value to use it later to add back to the main path.

    X_shortcut = X

    # if condition if reduce is True

    if reduce:

        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path

        # to do that, we need both CONV layers to have similar strides

        X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s))(X_shortcut)

        X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

        # if reduce, we will need to set the strides of the first conv to be similar to the shortcut strides

        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'valid')(X)

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

    else:

        # First component of main path

        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)

        X = BatchNormalization(axis = 3)(X)

        X = Activation('relu')(X)

        

    # Second component of main path

    X = Conv2D(filters = F2, kernel_size = kernel_size, strides = (1,1), padding = 'same')(X)

    X = BatchNormalization(axis = 3)(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)

    X = BatchNormalization(axis = 3)(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    return X
def ResNet50(input_shape, classes):

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    # Stage 1

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X_input)

    X = BatchNormalization(axis=3, name='bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = bottleneck_residual_block(X, 3, [64, 64, 256], reduce=True, s=1)

    X = bottleneck_residual_block(X, 3, [64, 64, 256])

    X = bottleneck_residual_block(X, 3, [64, 64, 256])



    # Stage 3

    X = bottleneck_residual_block(X, 3, [128, 128, 512], reduce=True, s=2)

    X = bottleneck_residual_block(X, 3, [128, 128, 512])

    X = bottleneck_residual_block(X, 3, [128, 128, 512])

    X = bottleneck_residual_block(X, 3, [128, 128, 512])



    # Stage 4

    X = bottleneck_residual_block(X, 3, [256, 256, 1024], reduce=True, s=2)

    X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    X = bottleneck_residual_block(X, 3, [256, 256, 1024])



    # Stage 5

    X = bottleneck_residual_block(X, 3, [512, 512, 2048], reduce=True, s=2)

    X = bottleneck_residual_block(X, 3, [512, 512, 2048])

    X = bottleneck_residual_block(X, 3, [512, 512, 2048])



    # AVGPOOL

    X = AveragePooling2D((1,1))(X)



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    # Create the model

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
 # min_lr: lower bound on the learning rate

# factor: factor by which the learning rate will be reduced

reduce_lr= ReduceLROnPlateau(monitor='val_loss',factor=np.sqrt(0.1),patience=5, min_lr=0.5e-6)



sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
# create model

model = ResNet50(input_shape = (112, 112, 3), classes = nr_foods)



# compile the model

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

 

# train the model

# call the reduce_lr value using callbacks in the training method

history = model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=10, verbose=0, callbacks=[reduce_lr])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()