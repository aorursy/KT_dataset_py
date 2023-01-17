import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
x_l = np.load('../input/X.npy')

y_l = np.load('../input/Y.npy')

img_size = 64

plt.subplot(1,2,1)

plt.imshow(x_l[260].reshape(img_size,img_size))

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis("off")
x_l.shape
y_l.shape
x = np.concatenate((x_l[204:409], x_l[822:1027]), axis = 0)

z = np.zeros(205)

o = np.ones(205)

y = np.concatenate((z,o), axis = 0).reshape(x.shape[0],1)
x.shape
y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state = 42)
x_train.shape
x_test.shape
x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

print(x_train_flatten.shape)

print(x_test_flatten.shape)
x_train = x_train_flatten.T

x_test = x_test_flatten.T

y_train = y_train.T

y_test = y_test.T

print("x train ", x_train.shape)

print("x test ", x_test.shape)

print("y train ", y_train.shape)

print("y test ", y_test.shape)
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    return cost
def forward_backward_propagation(w,b,x_train,y_train):

    

    #forward propagation begin.

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1 - y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    #forward propagation end.

    

    #backward propagation begin

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    return cost, gradients

    #backward propagation end   
def update(w,b,x_train,y_train, learning_rate, number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_of_iteration):

        

        cost, gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w-learning_rate * gradients["derivative_weight"]

        b = b-learning_rate * gradients["derivative_bias"]

        

        if i %10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

            

    parameters = {"weight": w, "bias":b}

    plt.plot(index, cost_list2)

    plt.xticks(index, rotation="vertical")

    plt.xlabel("Number of iteration")

    plt.ylabel("cost")

    plt.show()

    return parameters, gradients, cost_list

            
def predict(w,b,x_test):

    y_head = sigmoid(np.dot(w.T, x_test)+b)

    y_pred = np.zeros((1,x_test.shape[1]))

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            y_pred[0,i] = 0

        else:

            y_pred[0,i] = 1

    return y_pred
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    dimension = x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w,b,x_train, y_train, learning_rate, num_iterations)

    y_pred_test = predict(parameters["weight"], parameters["bias"], x_test)

    y_pred_train = predict(parameters["weight"], parameters["bias"], x_train)

    

    print("train accuracy : {} %".format(100-np.mean(np.abs(y_pred_train - y_train))*100 ))

    print("test accuracy : {} %".format(100-np.mean(np.abs(y_pred_test - y_test))*100 ))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 150)
from sklearn import linear_model

model = linear_model.LogisticRegression(random_state= 42, max_iter = 150)

print("test_acc: {}".format(model.fit(x_train.T, y_train.T).score(x_test.T,y_test.T)))

print("train_acc: {}".format(model.fit(x_train.T, y_train.T).score(x_train.T,y_train.T)))