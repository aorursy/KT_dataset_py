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
x_l=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")
Y_l=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")
img_size=64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))
plt.axis("off")
X = np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)
print(X.shape[1])
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y , test_size=0.15 , random_state = 42)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

x_train_flatten = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
x_test_flatten =  X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
print("x_train_flatten",x_train_flatten.shape)
print("x_test_flatten",x_test_flatten.shape)
x_train = x_train_flatten.T
x_test = x_train_flatten.T
y_train = Y_train.T
y_test = Y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0
    
    return w , b



def sigmoid(z):
    
    y_head = 1/(1+np.exp(-z))
    
    
    return y_head
y_head = sigmoid(0)
y_head

def forward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss =  -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1] 
    
    derivative_weight = (np.dot(x_train,(y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {"derivative_weight" : derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients
def update(w,b,x_train,y_train,learning_rate,num_iterations):
    
    cost_list = []
    
    cost_list2 = []
    
    index = []
    
    for i in range(num_iterations):
        
        cost,gradients = forward_propagation(w,b,x_train,y_train)
       
        
        w = w - learning_rate*gradients["derivative_weight"]
        
        b = b - learning_rate*gradients["derivative_bias"]
        
        cost_list.append(cost)
        
        if i % 10 == 0 :
            
            cost_list2.append(cost)
            
            index.append(i)
            
            
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters = { "weight" : w ,"bias" : b}  
        
    plt.plot(index ,cost_list2)
    plt.xticks(index ,rotation ='vertical')
    plt.xlabel('Number of iteration')
    plt.ylabel(" bilmiyim")
        
    plt.show()
        
    return parameters, gradients, cost_list
Y_prediction = np.zeros((1,x_train[1]))
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
# predict(parameters["weigh
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate ,num_iterations):
    
    
    dimension = x_train.shape[0]
    
    w,b = initialize_weights_and_bias(dimension)
    
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"] ,parameters ["bias"],x_test )
    
    print(" doğrulamaların yüzesini bulmak %{}".format(100 -np.mean(np.abs(y_prediction_test - y_train))*100))
    
    
    
    
    
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)