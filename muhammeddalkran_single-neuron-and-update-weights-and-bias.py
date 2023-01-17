# This codes are taken from datai team

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load data set

x_l = np.load('../input/sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(x_l[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis('off')

# Then lets create x_train, y_train, x_test, y_test arrays

# Join a sequence of arrays along an row axis.

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)

print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def function_name(first_parameter, second_parameter):

    sumation = first_parameter + second_parameter

    return sumation

sumation_result = function_name(4,5) # 4 + 5 = 9

print("sumation result is " + str(sumation_result))
def parameter_initialization(size):

    w = np.full((size,1),0.02)

    print(w.shape)

    b = 0.0

    return w,b

w,b = parameter_initialization(4096)


def sigmoid(z):

    y_head = 1/(np.exp(-z)+1)

    return y_head

#y_head = sigmoid(3)

#print(y_head) # for example as z = 3 result is ~0.952574; as z = 0 result is 0.5
def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/ x_train.shape[1]

    return cost, y_head
def backward_propagation(w,b,x_train,y_train):

    #forward propagation and return result cost and y_head

    #we use y_head in derivative of bias and weights.

    #we return cost and gradients, actually it is dictionary as data type, to update weight and bias



    cost, y_head = forward_propagation(w,b,x_train,y_train)

    sample_size = x_train.shape[1]

    weights_derivative = (np.dot(x_train,((y_head-y_train).T)))/sample_size

    bias_derivative = np.sum(y_head-y_train)/sample_size

    gradients = {"weights_derivative": weights_derivative, "bias_derivative":bias_derivative}

    return cost, gradients
def update_weights_and_bias(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(number_of_iteration):

        #backward and forward propagation

        cost,gradients = backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate*gradients["weights_derivative"]

        b = b - learning_rate*gradients["bias_derivative"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

        #update weights and bias

    parameters_dictionary = {"weights":w, "bias" : b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation = 'vertical')

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters_dictionary, gradients, cost_list
def prediction(w,b,x_test):

    z = np.dot(w.T,x_test) + b

    y_head = sigmoid(z)

    #for allacotion to make code execution faster. 

    #Time complexity is constant 0.

    #If we implement list, time complexity become n, not constant. 

    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):

    size = x_train.shape[0]

    w,b = parameter_initialization(size)

    parameters,gradients,cost_list = update_weights_and_bias(w,b,x_train,y_train,learning_rate,number_of_iteration)

    y_prediction_test = prediction(parameters["weights"],parameters["bias"],x_test)

    y_prediction_train = prediction(parameters["weights"],parameters["bias"],x_train)

    

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.02,number_of_iteration = 150)
from sklearn import linear_model as lm

logistic_regression = lm.LogisticRegression(random_state = 42, max_iter = 150)

print("train accuracy: {} %".format(logistic_regression.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))

print("test accuracy: {} %".format(logistic_regression.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))


