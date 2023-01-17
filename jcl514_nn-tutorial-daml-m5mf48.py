# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import scipy as sp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.linalg import norm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import copy

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def relu(x):

    return x*(np.sign(x)+1.)/2.
def sigmoid(x):

    return 1./(1.+np.exp(-x))
def softmax(x):

    return np.exp(x)/sum(np.exp(x))
def mynorm(Z):

    return np.sqrt(np.mean(Z**2))
def myANN(Y,Xtrain,Xpred,W01,W02,W03,b01,b02,b03):

    # Initialization of Weights and Biases

    W1 = copy.copy(W01)

    W2 = copy.copy(W02)

    W3 = copy.copy(W03)

    b1 = copy.copy(b01)

    b2 = copy.copy(b02)

    b3 = copy.copy(b03)

    # Initialize adhoc variables

    k = 1

    change = 999

    # Begin Feedforward

    while (change > 0.001 and k<201):

        print("Iteration", k)

        # Hidden Layer 1

        h1 = relu(W1@Xtrain + b1)

        # Hidden Layer 2

        h2 = sigmoid(W2@h1 + b2)

        # Output Layer

        Yhat = softmax(W3@h2 + b3)

        # Find cross-entropy loss

        loss = -Y@np.log(Yhat)

        print("Current Loss:",loss)

        

        # Find gradient of loss with respect to the weights

        # Output Later

        db3 = Yhat - Y # sum(Y)*Yhat - Y

        dW3 = np.outer(db3,h2)

        # Hidden Layer 2

        db2 = (W3.T@(db3))*h2*(1-h2) #((sum(Y)*W3.T@Yhat)-(W3.T@Y))*h2*(1-h2)

        dW2 = np.outer(db2,h1)

        # Hidden Layer 1

        db1 = np.sign(W1@Xtrain + b1)*(W2.T@(db2))

        dW1 = np.outer(db1,Xtrain)

        

        # Update Weights by Back Propagation

        # Output Layer

        b3 -= db3

        W3 -= dW3

        # Hidden Layer 2

        b2 -= db2

        W2 -= dW2

        # Hidden Layer 1

        b1 -= db1

        W1 -= dW1

        

        change = norm(db1)+norm(db2)+norm(db3)+norm(dW1)+norm(dW2)+norm(dW3)

        k+= 1

        

    h1pred = W1@Xpred + b1

    h2pred = W2@relu(h1pred) + b2

    opred = W3@sigmoid(h2pred) + b3

    Ypred = softmax(opred)

    print("")

    print("Summary")

    print("Target Y", Y)

    print("Fitted Ytrain", Yhat)

    print("Xpred", Xpred)

    print("Fitted Ypred", Ypred)

    print("Weight Matrix 1", W1)

    print("Bias Vector 1", b1)

    print("Weight Matrix 2", W2)

    print("Bias Vector 2", b2)

    print("Weight Matrix 3", W3)

    print("Bias Vector 3", b3)
W0_1 = np.array([[0.1,0.3,0.7],[0.9,0.4,0.4]])

b_1 = np.array([1.,1.])

W0_2 = np.array([[0.4,0.3],[0.7,0.2]])

b_2 = np.array([1.,1.])

W0_3 = np.array([[0.5,0.6],[0.6,0.7],[0.3,0.2]])

b_3 = np.array([1.,1.,1.])

YY = np.array([1.,0.,0.])

X_train = np.array([0.1,0.7,0.3])

X_pred = X_train
myANN(YY,X_train,X_pred,W0_1,W0_2,W0_3,b_1,b_2,b_3)
from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers
# Create Model

model = Sequential()

model.add(Dense(2, input_dim=3, activation='relu', weights = [W0_1.T,b_1]))

model.add(Dense(2, activation='sigmoid', weights = [W0_2.T,b_2]))

model.add(Dense(3, activation='softmax', weights = [W0_3.T,b_3]))

# Compile Model

sgd = optimizers.SGD(lr=1)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_crossentropy'])

model.get_weights()
# Fit the model

model.fit(X_train.reshape((1,3)), YY.reshape((1,3)), epochs=200, batch_size=1)
model.predict(X_pred.reshape((1,3)))
model.get_weights()