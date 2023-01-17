import numpy as np
import pandas as pd
import os, shutil
from keras.models import Model, load_model
from keras.layers import (Input, Concatenate, Dense, 
                          Flatten, Dropout, Conv2D, 
                          AveragePooling2D, GaussianNoise,
                          BatchNormalization)
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import keras.backend as K
'''
    Constucts single DenseNet block
    
    @param inputs - input 4D tensor
    @param n - number of dense units in block
    @param k - number of dense unit filters
    @param b - bottleneck factor of dense unit
'''
def DenseBlock(inputs, n = 4, k = 12, b = 2):
    for i in range(n):
        cur_layer = Conv2D(
            filters = b * k,
            kernel_size = (1, 1),
            activation = 'relu'
        )(inputs)
        cur_layer = Conv2D(
            filters = k,
            padding = 'same',
            kernel_size = (3, 3),
            activation = 'relu'
        )(cur_layer)
        cur_layer = Dropout(0.2)(cur_layer)
        inputs = Concatenate(
            axis = 3
        )([inputs, cur_layer])
    return inputs

'''
    Constucts DenseNet
'''
def DenseNet(k = 12):
    inputs = Input(shape = [28, 28, 1])
    cur_layer = Conv2D(
        filters = 2 * k,
        kernel_size = (5, 5),
        activation = 'relu'
        #kernel_regularizer = l2(0.05)
    )(inputs)
    cur_layer = AveragePooling2D(
        pool_size = (2, 2)
    )(cur_layer)
    # block 1
    cur_layer = DenseBlock(cur_layer, 4, k, 4)
    cur_layer = BatchNormalization()(cur_layer)
    cur_layer = Conv2D(
        filters = int(cur_layer.get_shape().as_list()[-1] / 2),
        kernel_size = (1, 1),
        activation = 'relu'
    )(cur_layer)
    cur_layer = AveragePooling2D(
        pool_size = (2, 2)
    )(cur_layer)
    # block 2
    cur_layer = DenseBlock(cur_layer, 8, k, 4)
    cur_layer = BatchNormalization()(cur_layer)
    cur_layer = Conv2D(
        filters = int(cur_layer.get_shape().as_list()[-1] / 2),
        kernel_size = (1, 1),
        activation = 'relu'
    )(cur_layer)
    cur_layer = AveragePooling2D(
        pool_size = (2, 2)
    )(cur_layer)
    # block 3
    cur_layer = DenseBlock(cur_layer, 8, k, 4)
    cur_layer = BatchNormalization()(cur_layer)
    cur_layer = Conv2D(
        filters = int(cur_layer.get_shape().as_list()[-1] / 2),
        kernel_size = (1, 1),
        activation = 'relu'
    )(cur_layer)
    cur_layer = AveragePooling2D(
        pool_size = (3, 3)
    )(cur_layer)
    # final
    cur_layer = Flatten()(cur_layer)
    cur_layer = Dense(
        units = 10,
        activation = 'softmax'
    )(cur_layer)
    model = Model(
        inputs = inputs,
        outputs = cur_layer
    )
    return model
train_data = pd.read_csv('../input/train.csv', delimiter=',')
X_train = np.reshape(train_data.loc[:, 'pixel0':].as_matrix(), [-1, 28, 28, 1]) / 255.0
y_train = train_data.loc[:, 'label']
y_train = to_categorical(y_train.as_matrix())
test_data = pd.read_csv('../input/test.csv', delimiter=',')
X_test = np.reshape(test_data.loc[:, 'pixel0':].as_matrix(), [-1, 28, 28, 1]) / 255.0

model = DenseNet(k = 24)
opt = SGD(
    lr = 0.1,
    decay = 1e-04,
    momentum = 0.9,
    nesterov = True
)
model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = opt, 
    metrics = ['categorical_accuracy']
)
model.fit(
    x = X_train, 
    y = y_train, 
    epochs = 5, 
    batch_size = 128, 
    verbose = 1
)
preds = model.predict(X_test)