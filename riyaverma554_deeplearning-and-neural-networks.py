# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x_l=np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')
y_l=np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')
img_size=64
plt.subplot(1,2,1)
plt.imshow(x_l[204].reshape(img_size,img_size))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_l[1027].reshape(img_size,img_size))
plt.axis('off')


X=np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)

z=np.zeros(205)
o=np.ones(205)
Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1)

print('X.shape: ',X.shape)
print('Y.shape: ',Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.15,random_state=42)
number_of_train=X_train.shape[0]
number_of_test=X_test.shape[0]
X_train_flatten= X_train.reshape(number_of_train, X_train.shape[1]* X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test, X_test.shape[1]* X_test.shape[2])

print('X_train_flatten: ',X_train_flatten.shape)
print('X_test_flatten: ',X_test_flatten.shape)


x_train=X_train_flatten.T
x_test=X_test_flatten.T
y_train=Y_train.T
y_test=Y_test.T

print('x_train: ',x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
def Initialize_weights_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    y_head
def Forward_propagation(w, b, x_train, y_train):
    z=np.dot((w.T,x_train)+b)
    y_head=sigmoid(z)
    loss= -y_train * np.log(y_head) - (1-y_train)* np.log(1-y_head)
    cost= np.sum(loss)/x_train.shape[1]
    
    return cost
def forward_backward_propagation(w,b,x_train,y_train):
    z=np.dot((w.T,x_train)+b)
    y_head=sigmoid(z)
    loss= -y_train * np.log(y_head) - (1-y_train)* np.log(1-y_head)
    cost= np.sum(loss)/x_train.shape[1]
    
    derivative_weight= np.dot(x_train, ((y_head-y_train).T))/x_train.shape[1]
    derivative_bias= np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients= {'derivative weights': derivative_weight,'derivative_bias': derivative_bias}
    
    return cost,gradients
    
def update(w, b, x_train, y_train, learning_rate, number_of_iterations):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    for i in range(number_of_iterations):
        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients['derivative_weights']
        b = b - learning_rate * gradients['derivative_bias']
        
        if i%10==0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters={'weights': w, 'bias': b}
    plt.plot(index,cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('number of iterations')
    plt.ylabel('cost')
    plt.show()
    
    return parameters, gradients, cost_list
    
from sklearn import linear_model
log_reg=linear_model.LogisticRegression(random_state=42, max_iter=150)
print('test accuracy: {}'.format(log_reg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))
print('train accuracy: {}'.format(log_reg.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))
#ARTIFICIAL NEURAL NETWORKS
def initialize_parameters_and_layerSizesNN(x_train,y_train):
    parameters={'weight1': np.random.rand(3,x_train.shape[0]) * 0.1,
               'bias1': np.zeros((3,1)),
               'weight2': np.random.randn(3,y_train.shape[0]) * 0.1,
               'bias2': np.zeros((y_train.shape[0],1))}
    return parameters

def forward_propagationNN(x_train, parameters):
    Z1= np.dot(parameters['weight1'],x_train)+ parameters['bias1']
    A1=np.tanh(Z1)
    Z2=np.dot(parameters['weight2'], x_train)+ parameters['bias2']
    A2=sigmoid(Z2)
    
    cache={'Z1':Z1,
          'A1':A1,
          'Z2':Z2,
          'A2':A2}
    return A2,cache

def computecostNN(A2, Y, parameters):
    logprobs=np.multiply(np.log(A2),Y)
    cost= -np.sum(logprobs)/Y.shape[1]
    return cost
def backpropagationNN(parameters, cache, X, Y):
    dZ2= cache['A2']-Y
    dW2= np.dot(dZ2, cache['A1'].T)