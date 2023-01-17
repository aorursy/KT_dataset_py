# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def getonehot(num):
    a = np.zeros(10)
    a[num] = 1
    return a 
def resetonehot(a):
    return a.argmax()
train_data = pd.read_csv("../input/train.csv")
train_data_values = train_data.values
train_data_values = train_data_values.T
#m_train = train_data_values.shape[1]

test_data = pd.read_csv("../input/test.csv")
test_data_values = test_data.values
test_data_values = test_data_values.T
m_test = test_data_values.shape[1]

#x_train,y_train = train_data_values[2:,:], np.array(train_data_values[1,:]).reshape(1,m_train)
x_train,y_train_pre = train_data_values[:-1,0:42000], np.array(train_data_values[0,0:42000]).reshape(1,42000)
m_train = x_train.shape[1]

y_train = np.zeros((10,m_train))
j = 0
for i in y_train_pre[0]:
    y_train[:,j] = getonehot(i)
    j = j+1
    
x_test = test_data_values[:,:]
n = x_test.shape[0]
y_test = np.zeros((10,m_test))

print("Shape of x_train is ",x_train.shape)
print("Shape of y_train is ",y_train.shape)
print("Shape of x_test is ",x_test.shape)
print("Shape of y_test is ",y_test.shape)
print("number of features is ",n)

a = np.average(x_train,axis=0).reshape(1,m_train)
std = np.std(x_train,axis=0).reshape(1,m_train)
x_train = (x_train - a)/std

a = np.average(x_test,axis=0).reshape(1,m_test)
std = np.std(x_test,axis=0).reshape(1,m_test)
x_test = (x_test - a)/std
def sigmoid(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid
def relu(x):
    return x * (x > 0)
def forward_backward_propogation(X,y,w,b):

    z = np.dot(w.T,X) + b
    ycap= sigmoid(z)

    #calculate loss & cost
    loss = -y*np.log(ycap)-(1-y)*np.log(1-ycap)    
    cost = np.sum(loss)/m_train

    dw = (1/m_train) * np.dot (X,(ycap - y).T)
    db = (1/m_train) * np.sum(ycap - y)
    
    gradients = {"dw":dw, "db":db}
    
    return cost, gradients
def update_params(X,y,w,b, learning_rate, num_iter):
    cost_list = []
    for i in range(num_iter):
        cost, gradients = forward_backward_propogation(X,y,w,b)
        cost_list.append(cost)
        print ("iteration ",i,":","cost :",cost)
        w = w - learning_rate * gradients["dw"]
        b = b - learning_rate * gradients["db"]
    parameters = {"w":w,"b":b}
    
    return parameters, gradients
def predict(w, b, X):
    z = sigmoid(np.dot(w.T,X)+b)

    for i in range(z.shape[1]):
        temp = z[:,i]

    Y_pred = np.zeros((10,X.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i,j]<= 0.5:
                Y_pred[i,j] = 0
            else:
                Y_pred[i,j] = 1
    return Y_pred
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    #Initializing parameters
    w = np.random.randn(n,10) * 0.01
    #w = np.zeros((n,10))
    b = 0.0
    
    parameters, gradients = update_params(x_train, y_train, w, b, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["w"],parameters["b"],x_test)
    y_prediction_train = predict(parameters["w"],parameters["b"],x_train)

    y_pred_post = []
    for i in range(y_prediction_train.shape[1]):
        temp = y_prediction_train[:,i]
        y_pred_post.append(resetonehot(temp.reshape(1,10)))
    
    # Print train/test Errors
    print("train accuracy: ",(100/m_train)*(np.count_nonzero(y_pred_post - y_train_pre == 0)),"%")
    
    y_pred_test = []
    for i in range(y_prediction_test.shape[1]):
        temp = y_prediction_test[:,i]
        pred = resetonehot(temp.reshape(1,10))
        print(i,",",pred)
    
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.17, num_iterations = 2000)
