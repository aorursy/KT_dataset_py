# to do OHE
from keras.utils import to_categorical
# the only python lib we really need
import numpy as np
# Keras tools
from keras.models import Sequential
from keras.layers import Dense, Flatten
# read data
import pandas as pd
mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
# do numpy arrays
Xtrain = mnist_train.drop(['label'], axis=1).values
Ytrain =  mnist_train.loc[:, 'label'].values
Xtest = mnist_test.drop(['label'], axis=1).values
Ytest =  mnist_test.loc[:, 'label'].values
Xtrain = Xtrain.reshape(60000, 28, 28)
Xtest = Xtest.reshape(10000, 28, 28)
Xtrain.shape
# one hot encoded Y
Ytrain_ohe = to_categorical(Ytrain) 
Ytest_ohe = to_categorical(Ytest) 
# new option - scalling to 0-1
Xtrain = Xtrain / 255.0
Xtest  = Xtest / 255.0
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(300, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(10,  activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(Xtrain, Ytrain_ohe, epochs=10, batch_size=32)
model.evaluate( Xtest, Ytest_ohe )