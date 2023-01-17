# to do OHE labels

from keras.utils import to_categorical

# to show numbers

import matplotlib.pyplot as plt

# the only python lib we really need

import numpy as np
# read data

import pandas as pd

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
mnist_train.head()
# do numpy arrays

Xtrain = mnist_train.drop(['label'], axis=1).values

Ytrain =  mnist_train.loc[:, 'label'].values

Xtest = mnist_test.drop(['label'], axis=1).values

Ytest =  mnist_test.loc[:, 'label'].values
print(Xtrain.shape)

print(Ytrain.shape)

print(Xtest.shape)

print(Ytest.shape)
number_example = Xtrain[0].reshape(28, 28)
plt.imshow(number_example, cmap='gray')
# one hot encoded Y

Ytrain_ohe = to_categorical(Ytrain) 

Ytest_ohe = to_categorical(Ytest) 

print(Ytrain.shape)

print(Ytrain_ohe.shape)

print(Xtest.shape)

print(Ytest_ohe.shape)

print(Ytrain[0])

print(Ytrain_ohe[0])
# layers size

input_layer = 784

hidden_layer = 100

output_layer = 10
# initial weights and bias

W1 = np.random.randn( input_layer, hidden_layer ) #W1

b1 = np.random.randn( 1, hidden_layer ) #b1

W2 = np.random.randn( hidden_layer, output_layer ) #W2

b2 = np.random.randn( 1, output_layer ) #b2
W1.shape
# activations functions

def tanh(x):

    return np.tanh(x)



# for tyhe last layer (output)

def softmax(x):

    exp_scores = np.exp(x)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
# first layer

Z1 = Xtrain.dot(W1) + b1

print(Z1.shape)
# first layer after activation

A1 = tanh(Z1)

print(A1.shape)
# second layer

Z2 = A1.dot(W2) + b2

print(Z2.shape)
# second layer after activation - networkoutput (yhat)

A2 = softmax(Z2)

print(A2.shape)
# predict 

predictions = np.argmax(A2, axis=1)

print(predictions.shape)
# forward propagation

def forward(X):

    Z1 = X.dot(W1) + b1

    A1 = tanh(Z1)

    Z2 = A1.dot(W2) + b2

    A2 = softmax(Z2)

    return A2
def loss(y, X):

    N = len(y)

    yhat = forward(X)

    logs = np.sum(np.log(yhat[range(N), y]))

    return -1.0/N * logs
def loss_alt(y, X):

    yhat = forward(X)

    return - np.mean( np.log( yhat[ range(len(yhat)), y ] ) )
# check our loss

print(loss(Ytrain, Xtrain))

print(loss_alt(Ytrain, Xtrain))
# tanh derivative

def tanh_dev(x):

    return 1.0-np.tanh(x)**2
# learning rate

learning_rate = 0.1
#backpropagation

# delta 2

delta2 = (A2-Ytrain_ohe)/len(Ytrain)



# to compute delta1 we need 

dZ1 = tanh_dev(Z1)



# delta 1

delta1 = delta2.dot(W2.T) * dZ1



# partial derivatives for weighs

dev_W2 = A1.T.dot(delta2)

dev_W1 = Xtrain.T.dot(delta1)



# partial derivatives for bias

dev_b2 = np.sum( delta2, axis=0, keepdims=True )

dev_b1 = np.sum( delta1, axis=0, keepdims=True )



# update waights and bias

W1 -= (learning_rate * dev_W1)

b1 -= (learning_rate * dev_b1)

W2 -= (learning_rate * dev_W2)

b2 -= (learning_rate * dev_b2)
print(loss(Ytrain, Xtrain))