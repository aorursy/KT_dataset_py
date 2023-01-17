from keras.models import Sequential

from keras.layers import Dense, Input

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras import backend as K

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pandas as pd 





data = pd.read_csv('../input/A_Z Handwritten Data/A_Z Handwritten Data.csv')
print(data.shape) 



data.rename(columns={'0':'label'}, inplace=True)



print(data.head())



X = data.drop('label',axis = 1)

y = data['label']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler

standard_scaler = MinMaxScaler()

standard_scaler.fit(X_train)



X_train = standard_scaler.transform(X_train)

X_test = standard_scaler.transform(X_test)
print(X_train.shape)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')



Y_train = np_utils.to_categorical(Y_train)

Y_test = np_utils.to_categorical(Y_test)



print(Y_test.shape) # (122909, 26)



num_classes = Y_test.shape[1]
# Lenet Model

def mymodel(input_shape):

    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((2,2))(X_input)



    # CONV -> RELU Block applied to X

    X = Conv2D(6, (5,5), strides = (1, 1), name = 'conv0')(X)

    X = Activation('relu')(X)

    X = Dropout(0.2)(X)

    

    #pooling

    X = MaxPooling2D(pool_size=2, strides=2, name='max_pool0')(X)

    

    # CONV -> RELU Block applied to X

    X = Conv2D(6, (5,5), strides = (1, 1), name = 'conv1')(X)

    X = Activation('relu')(X)

    

    #pooling

    X = MaxPooling2D(pool_size=2, strides=2, name='max_pool1')(X)

    

    X = Dropout(0.2)(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(120, activation='relu', name='fc1')(X)

    X = Dense(84, activation='relu', name='fc2')(X)

    

    X = Dense(num_classes, activation='softmax')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='mymodel')

    



    

    return model
from keras.layers import ZeroPadding2D,Activation

from keras.models import Model

MyModel = mymodel((28,28,1))
MyModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
MyModel.fit(x = X_train, y = Y_train, epochs = 1, batch_size = 64)
preds = MyModel.evaluate(x = X_test, y = Y_test)
print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))