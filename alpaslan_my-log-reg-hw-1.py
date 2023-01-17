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
# Read csv
data = pd.read_csv("../input/voice.csv")
data.label = [1 if each == "male" else 0 for each in data.label]
print(data.info())

y = data.label.values
x_data = data.drop(["label"],axis=1)    #label yokkenki hali
#%% Normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# ( x - min(x) ) / ( max(x) - min(x) )
#%% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
# %% parameeter initialize and sigmoid function

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)     # [dimension,1]'lik arrayda her kutuya 0.01 yaz
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# print(sigmoid(0))
# %%
def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b     # matris çarpımı ==> w.T=(1,20) & x_train=(20,2534)
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]     # x_train.shape[1] is for scaling = türevlerini aldık
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]       # x_train.shape[1] is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients       # gradients = parametrelerin depolandığı dictionary

# %% Updating (learning) parameters

 # learning_rate = ne kadar hızlı öğrendiğim. büyük değer olursa doğru bir şekilde öğrenememem ihtimali var

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []         
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:     # 10 adımda bir cost'u göster
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    
    # We update(learn) parameters weight and bias
    parameters = {"weight": w, "bias": b}   # w ve b yi güncellemek train etmek demek
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list         # gradients ==> en son türevli halleri

# %%

def predict(w, b, x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1)
    # if z is smaller than 0.5, our prediction is sign zero(y_head=0)
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction
# %% logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]    # that is 20
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    
    # print test Error
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, num_iterations = 200)

#%% sklearn with LR

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)

print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))
