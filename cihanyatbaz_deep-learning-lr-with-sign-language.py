# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load data set
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64  # pixel size

 # for sign zero
plt.subplot(1,2,1)  
plt.imshow(x_l[260])  # Get 260th index
plt.axis("off")

# for sign one
plt.subplot(1,2,2)
plt.imshow(x_l[900])  # Get 900th index
plt.axis("off")
# From 0 to 204 zero sign, from 205 to 410 is one sign
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)

# We will create their labels. After that, we will concatenate on the Y.
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o), axis=0).reshape(X.shape[0],1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)
# random_state = Use same seed while randomizing
print(x_train.shape)
print(y_train.shape)
x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print('x_train_flatten: {} \nx_test_flatten: {} '.format(x_train_flatten.shape, x_test_flatten.shape))
# Here we will change the location of our samples and features. '(328,4096) -> (4096,328)' 
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
# Now let's create the parameter and sigmoid function. 
# So what we need is dimension 4096 that is number of pixel as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b

# Sigmoid function
# z = np.dot(w.T, x_train) +b
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))  # sigmoid function finding formula
    return y_head
sigmoid(0)  # o should result in 0.5
w,b = initialize_weights_and_bias(4096)
print(w)
print("----------")
print(b)
# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w, b, x_train, y_train):
    # Forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)
    cost = (np.sum(loss)) / x_train.shape[1]   # x_train.shape[1] is for scaling
    
    # Backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] # x_train.shape[1] is for 
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # x_train.shape[1] is for     
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)  # adding costs to cost_list
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]        
        if i % 10 == 0:
            cost_list2.append(cost)   # adding costs to cost_list2
            index.append(i) # Adds a cost to the index in every 10 steps
            print("Cost after iteration %i: %f" %(i, cost))
        
        # we update (learn) parameters weights and bias
    
    parameters = {"weight": w, "bias":b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
# Let's create prediction parameter

def predict(w,b,x_test):
    # x_test is a input for forward prapagation
    z = sigmoid(np.dot(w.T, x_test) +b)
    y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head = 1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head = 0),      
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, n_iterations):
    # Initialize
    dimension = x_train.shape[0]   # that is 4096
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, n_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train) 
    
    # print train / test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))    

logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, n_iterations = 170)
from sklearn import linear_model
lr_sl = linear_model.LogisticRegression(random_state=42, max_iter = 150)

print("test accuracy: {}".format(lr_sl.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
