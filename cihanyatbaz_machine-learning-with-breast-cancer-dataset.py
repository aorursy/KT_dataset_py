# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Prepare to data
data = pd.read_csv("../input/breast-cancer.csv")
data.head()
data.info()
# Let's wipe some columns that we won't use
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)  #axis=1 tüm sütunu siler
data.head()
data.describe()
# Let's take the some columns we'll use for show data means
data_mean= data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',
                 'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
                 'symmetry_mean','fractal_dimension_mean']]
color_list = ['cyan' if i=='M' else 'orange' for i in data_mean.loc[:,'diagnosis']]
pd.plotting.scatter_matrix(data_mean.loc[:, data_mean.columns != 'diagnosis'],
                           c=color_list,
                           figsize= [15,15],
                           diagonal='hist',
                           alpha=0.5,
                           s = 200,
                           marker = '*',
                           edgecolor= "black")
                                        
plt.show()
# Values of 'Benign' and 'Malignant' cancer cells
sns.countplot(x="diagnosis", data=data)
data.loc[:,'diagnosis'].value_counts()
# Let's convert "male" to 1, "female" to 0 values
data.diagnosis = [ 1 if each == "M" else 0 for each in data.diagnosis]
data.info()
# Let's determine the values of y and x axes
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
# Now we are doing normalization. Because if some of our columns have very high values, they will suppress other columns and do not show much.
# Formulel : (x- min(x)) / (max(x) - min(x))
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x.head()
# Now we reserve 80% of the values as 'train' and 20% as 'test'.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

# Here we will change the location of our samples and features. '(455,30) -> (30,455)' 
x_train = x_train.T   
x_test = x_test.T
y_train = y_train.T   
y_test = y_test.T

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)
# Now let's create the parameter and sigmoid function. Videodan nedenini yaz
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0   # It will be float
    return w,b

# Sigmoid Function

# Let's calculating z
# z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z)) # sigmoid functions finding formula
    return y_head
sigmoid(0)  # 0 should result in 0.5
# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):
    
    # forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)  
    cost =(np.sum(loss))/x_train.shape[1]         # x_train.shape[1] for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] 
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients
# Now let's apply Updating Parameter

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # Updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost gradients
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
    parameters = {"weight":w, "bias":b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("cost")
    plt.show()
    return parameters, gradients, cost_list

# Let's create prediction parameter
def predict(w,b,x_test):
    # x_test is an input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1

    return y_prediction

#Logistic Regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 455
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print train/test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 100)
# We can increase the accuracy of the test by playing with learning_rate and num_iterations
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 5, num_iterations = 150)
from sklearn import linear_model
lgrg = linear_model.LogisticRegression(random_state=42, max_iter=150)

print("test accuracy: {} ".format(lgrg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))