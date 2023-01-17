# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.info()
data.head()
# Dependent variables, class label for train dataset

y=data["Outcome"].values  #array

y
# Independent variables, traindataset

x_data = data.drop(["Outcome"],axis=1)

x_data.head()
# Normalization Dataset

# normalized data = (X - X_MİN) / (X_MAX - X_MİN)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)



# Feature ve onlara ait değerlerin yeri değiştirildi.

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("Changed of Features and Values place.")



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# Initialize

# w = Create matrix array and all array values are 0.01

# np.exp = exponent method in numpy library

def initialize_weights_and_bias(dimension):

    #initialize

    w = np.full((dimension,1),0.01) 

    b=0.0

    return w,b



def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss =  -(1 - y_train) * np.log(1 - y_head) -y_train * np.log(y_head)

    cost = (np.sum(loss)) / x_train.shape[1]

    

    #backward propagation

    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] #derivative based on weight

    derivative_bias = np.sum(y_head - y_train ) / x_train.shape[1] #derivative based on bias

    

    #weight and bias are derivates kept in dictionary(gradients)

    gradients = {"derivative_weight": derivative_weight, "derivative_bias":derivative_bias}

    

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

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
# Predict Method



def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test) + b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    

    #initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    

    #update method for forward and backward propagation

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    



    # Print train/test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    
#test

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 5, num_iterations = 300)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("Test Accuracy : {}".format(lr.score(x_test.T,y_test.T)))
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))