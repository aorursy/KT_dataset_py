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
#importing dataset

data = pd.read_csv('../input/Prostate_Cancer.csv')
data.head()
data.info()
data.dropna(inplace = True)
#dropping id. id cannot be tought as feature which affects the result

data.drop(['id'], axis = 1, inplace = True)
# in order to apply logistic regression, results should be binary values

data.diagnosis_result = [1 if each == "M" else 0 for each in data.diagnosis_result]
data.info()
data.shape
#diagnosis_result is the value which will be predicted.

y = data.diagnosis_result.values

y.shape
#after dropping diagnosis results, we achieve feature set that gives the values 

#that is used in the model to predict results

x_data = data.drop(['diagnosis_result'], axis = 1)

x_data.shape
#normalization should be applied in order to balance all features

x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#we need to split dataset for train and test

#x_train part of the data will be trained and tested with x_test, and after 

#the results,

#y_train part of the data will be tested according to y_test part 

#this is to determine the performance of the model

from sklearn.model_selection import train_test_split

#train_test_split function gives 4 subsets we need:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

#the sets should have format (rows=feature_number:dimension, columns=data_number)

#so we took transpose of the four dataset:

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
#initializing parameters w and b

#dimension means number of features

def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b
#sigmoid function is applied to z.

#z is equal to sum of (all feature values * weights + bias)

#or z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head

#this function gives y_head -->> y_head gives a prediction of the result but 

#the results are not safe because w and b values are given intuitively.
#forward and backward propagation

#forward propagation is all the steps from features to cost calculation

#with first values of w and b, model gives prediction, it will be evaluated and 

#derivatives of the weight and bias is taken 

#in other words, our model needs to learn the parameters weights and bias that 

#minimize cost function. 

#This technique is called gradient descent.

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    #z value is calculated with x_train and initial w,b values

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) #z value is applied sigmoid function

    #loss is calculated for all rows of the dataset.

    #loss function gives us prediction results are bad or good

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) 

    #cost is the sum of all loss.

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
# Updating(learning) parameters

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

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    return parameters, gradients, cost_list

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
 # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    train_accuracy = (100 - np.mean(np.abs(y_prediction_train - y_train)) * 100)

    test_accuracy = (100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)

    return train_accuracy, test_accuracy
a = 0

for i in range(1,11):

    for j in range(1,11):

        train_accuracy = logistic_regression(x_train, y_train, x_test, y_test, learning_rate = i*0.1, num_iterations = j*50)[0]

        test_accuracy = logistic_regression(x_train, y_train, x_test, y_test, learning_rate = i*0.1, num_iterations = j*50)[1]

        num_iterations = j*50

        learning_rate = i*0.1

        if test_accuracy > a:

            a = test_accuracy

            b = train_accuracy

            c = learning_rate

            d = num_iterations

print('learning_rate: ', c,'num_iterations: ',d )

print('train_accuracy:',b, 'test_accuracy:', a)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 500)

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))