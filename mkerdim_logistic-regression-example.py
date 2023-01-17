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



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/deodorant-instant-liking-data/Data_train_reduced.csv")

test_data = pd.read_csv("/kaggle/input/deodorant-instant-liking-data/test_data.csv")



data.info()

test_data.info()
test_data.Product.unique()
print(test_data[test_data["Product"] == "Deodorant F"].shape)

print(test_data[test_data["Product"] == "Deodorant G"].shape)
filt_1 = test_data["Product"] == "Deodorant F"

filt_2 = test_data["Product"] == "Deodorant G"



my_data = test_data[filt_1 | filt_2].copy()
my_data.info()
my_data.drop(["Respondent.ID", "Product.ID"], axis=1, inplace=True)
my_data.Product = [1 if i == "Deodorant F" else 0 for i in my_data.Product]
y = my_data.Product.values

x = my_data.drop(["Product"],axis=1)

x_normalised = (x - np.min(x)) / (np.max(x) - np.min(x))
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_normalised, y, test_size=0.2, random_state=15)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
print(x_train.shape)

print(x_test.shape)



print(y_train.shape)

print(y_test.shape)
# initialise the w and b values

def set_initial_w_and_b_values(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b



# sigmoid function for making the values between 0 and 1

def sigmoid(z):

    return 1.0/(1.0 + np.exp(-z))



# forward and backward propagation

def forward_and_backward_propagation(w,b,x,y):

    z = np.dot(w.T,x) + b

    y_head = sigmoid(z)

    

    loss = -y*np.log(y_head)-(1-y)*np.log(1-y_head)

    cost = (np.sum(loss))/x.shape[1]

    

    derivative_weight = (np.dot(x,((y_head-y).T)))/x.shape[1]

    derivative_bias = np.sum(y_head-y)/x.shape[1]

    

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost, gradients



# update the parameters according to gradients

def update(w, b, x, y, learning_rate, number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost, gradients = forward_and_backward_propagation(w,b,x,y)

        cost_list.append(cost)

        # update parameters

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % int(number_of_iterarion/20) == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i+1, cost))

            

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



# to make predictions

def predict(w,b,x):

    z = sigmoid(np.dot(w.T,x)+b)

    Y_prediction = np.zeros((1,x.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction



# logistic regression function

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]

    w,b = set_initial_w_and_b_values(dimension)



    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 300) 