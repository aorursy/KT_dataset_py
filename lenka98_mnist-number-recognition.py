# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.head(10)
data = train_df.values
data
train_data = data[:, 1:]/255
train_labels = data[:, 0]
train_data
train_labels
print(len(train_df.columns))
print(train_data.shape)
print(train_labels.shape)
train_data = train_data.reshape(42000, 28, 28, 1)
train_labels = train_labels.reshape(42000, 1)
print(train_data.shape)
print(train_labels.shape)
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.utils import to_categorical
train_labels = to_categorical(train_labels, num_classes = 10)
train_labels.shape
print(train_labels[:10])
def simpleCNNBlock (X, num_filters, f, block_num):
    
    X = Conv2D(filters = num_filters, 
               kernel_size = (f, f), 
               strides = (1, 1),
               padding = 'same',
               name = 'ConvLayer-' + str(block_num), 
               kernel_initializer = 'glorot_uniform')(X)
   
    X = BatchNormalization(axis = 3, 
                           name = 'BatchNormLayer-' + str(block_num))(X)
    
    X = Activation('relu')(X)
    
    X = MaxPooling2D(pool_size= (f, f), 
                     strides=(1, 1),
                     padding = 'valid', 
                     name = 'MaxPoolLayer-' + str(block_num))(X)
   
    return X
def simpleCNNModel (input_shape, classes):
    
    X_input = Input(input_shape)
    
    X = simpleCNNBlock(X_input, 16, 5, 1)
    X = simpleCNNBlock(X, 32, 5, 2)
    X = simpleCNNBlock(X, 64, 3, 3)
    X = simpleCNNBlock(X, 128, 3, 4)
    
    X = Conv2D(filters = 128, 
               kernel_size = (3, 3), 
               strides = (1, 1),
               padding = 'valid',
               name = 'ConvLayer-5', 
               kernel_initializer = 'glorot_uniform')(X)
    
    X = BatchNormalization(axis = 3, 
                           name = 'BatchNormLayer-5')(X)
    
    X = Activation('relu')(X)
    
    X = AveragePooling2D(pool_size= (3, 3), 
                         strides=(2, 2),
                         padding = 'valid', 
                         name = 'AvgPoolLayer')(X)
    X = Conv2D(filters = 32, 
               kernel_size = (1, 1), 
               strides = (1, 1),
               padding = 'valid',
               name = 'ConvLayer-6', 
               kernel_initializer = 'glorot_uniform')(X)
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = 'glorot_uniform')(X)
    
    model = Model(inputs = X_input, outputs = X, name='CNNModel')
    
    return model
model = simpleCNNModel(input_shape = (28, 28, 1), classes = 10)
model.compile(optimizer = Adam(lr = 0.001,
                               beta_1 = 0.9,
                               beta_2 = 0.999,
                               epsilon = 10e-8),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(train_data, train_labels, batch_size = 32, epochs = 2, validation_split = 0.2)
test_df = pd.read_csv('../input/test.csv')
test = test_df.values
test = test/255
test = test.reshape(-1, 28, 28, 1)
test.shape
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
