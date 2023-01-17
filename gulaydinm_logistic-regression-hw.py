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
data = pd.read_csv("../input/data.csv")
data.head()
data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
data.head()
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())
data.head()
y_raw = data.diagnosis.values 

# We should make inplace "false" otherwise Python will consider as an error. 

x_raw = data.drop(["diagnosis"] , axis=1 ,inplace = False) 

x_normalized = (x_raw - np.min(x_raw)/np.max(x_raw) - np.min(x_raw)).values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_normalized,y_raw,test_size = 0.2 , random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T

def weights_and_bias(dimension):

    w = np.full((dimension,1), 0.01)

    b = 0.0

    return w,b



def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head

def forward_backward_propagation(w,b,x_train,y_train):

    # **forward propagation**

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -(1-y_train)*np.log(1-y_head) - y_train*np.log(y_head)

    cost = (np.sum(loss)) / x_train.shape[1]

    # ***********************

    

    # **backward propagation**

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]      

    gradients = {"derivative_weight" : derivative_weight , "derivative_bias" : derivative_bias}

    # ***********************

    

    return cost,gradients
def update(w,b,x_train,y_train,learning_rate,num_of_iterations):

    cost_list = []

    cost_list_print = []

    index = []

    

    for i in range (num_of_iterations):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate*gradients["derivative_weight"]

        b = b - learning_rate*gradients["derivative_bias"]

        if (i%10 == 0):

            cost_list_print.append(cost)

            index.append(i)

            print("Cost after {} iteration : {}".format(i,cost)) 

        

    

    parameters = {"weight" : w , "bias" : b}

    plt.plot(index,cost_list_print)

    plt.xticks(index,rotation = 'vertical')

    plt.xlabel("Number of iterations")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters,gradients,cost_list

        
def predict(w,b,x_test):

    #In this case we will consider x_test as an input for forward propagation.

    

    z = sigmoid(np.dot(w.T,x_test) + b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range (z.shape[1]):

        if (z[0,i] <= 0.5):

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction        

        

        

    
def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,num_of_iterations):

    #We will define a dimension.

    dimension = x_train.shape[0]

    w,b = weights_and_bias(dimension)

    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,num_of_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    print("Test accuracy is {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    logistic_regression(x_train,x_test,y_train,y_test,learning_rate = 3,num_of_iterations = 600)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("Test accuracy is {}".format(lr.score(x_test.T,y_test.T)))
