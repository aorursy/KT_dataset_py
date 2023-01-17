# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# preparation

data=pd.read_csv("../input/Iris.csv")
data.info()
data.drop(["Id"],axis=1,inplace=True)
data=data[(data.Species=="Iris-setosa") | (data.Species=="Iris-versicolor")]
data.Species.unique()
data.Species=[1 if i=="Iris-setosa" else 0 for i in data.Species]
y=data.Species.values
x_=data.drop(["Species"],axis=1)
#normalization

x=(x_ - np.min(x_))/(np.max(x_)-np.min(x_)).values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T
def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





# w,b = initialize_weights_and_bias(30)



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

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

            print ("Cost after iteration {}.: {} ".format(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='90')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    y_head = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(y_head.shape[1]):

        if y_head[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    dimension =  x_train.shape[0] 

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations) 

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 200)    
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x,y)

print("Accuaracy {}".format(lr.score(x_test.T,y_test.T)))