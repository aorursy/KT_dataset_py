# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings 

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x_1 = np.load('../input/Sign-language-digits-dataset/X.npy')

y_1 = np.load('../input/Sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(x_1[260].reshape(img_size,img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(x_1[900].reshape(img_size, img_size))

plt.axis('off')
X=np.concatenate((x_1[204:409], x_1[822:1027]),axis=0)

z = np.zeros(205)

o = np.ones(205)

Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1)

print("X shape: ",X.shape)

print("Y shape: ",Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])

print(X_train_flatten)

print(X_train_flatten.shape)

print(X_test_flatten)

print(X_test_flatten.shape)
x_test = X_test_flatten.T

x_train = X_train_flatten.T

y_test = Y_test.T

y_train = Y_train.T

print("x test: ",x_test.shape)

print("x train: ",x_train.shape)

print("y test: ",y_test.shape)

print("y train: ",y_train.shape)
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b
def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head

def forward_propagation(w,b,x_train,y_train):

    z= np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = np.sum(loss)/x_train.shape[1]

    return cost
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients



def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            y_prediction[0,i]=0

        else:

            y_prediction[0,i]=1

    return y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    dimension = x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    from sklearn import linear_model

    logreg =linear_model.LogisticRegression(random_state=42,max_iter=150)

    print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

    print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))

    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=150)