# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
data_x=np.load('../input/Sign-language-digits-dataset/X.npy')

data_y=np.load('../input/Sign-language-digits-dataset/Y.npy')

img_size=64

plt.subplot(1,1,1)

plt.imshow(data_x[210].reshape(img_size,img_size))

plt.axis('off')
x=np.concatenate((data_x[204:409],data_x[822:1027]),axis=0)

z=np.zeros(205)

o=np.ones(205)

y=np.concatenate((z,o),axis=0).reshape(-1,1)

# print(x.shape)

# print(y.shape)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=5)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
x_train=x_train.reshape(x_train.shape[0],img_size*img_size)

x_test=x_test.reshape(x_test.shape[0],img_size*img_size)

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
def initialize_parameters(layer_dims):

    np.random.seed(1)

    L=len(layer_dims)

    parameters={}

    for l in range(1,L):

        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01

        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

    return parameters

        
# parameters={}

parameters=initialize_parameters([5,4,3])

print(str(parameters['W1'].shape)+'\n'+str(parameters['W1']))

print(str(parameters['b1'].shape)+'\n'+str(parameters['b1']))

print(str(parameters['W2'].shape)+'\n'+str(parameters['W2']))

print(str(parameters['b2'].shape)+'\n'+str(parameters['b2']))
def relu(Z):

#     A=[max(0,i) for i in x ]

    A=np.where(Z>0,Z,0)

    return A,Z

def sigmoid(Z):

    A=1/(1+np.exp(-Z))

    return A,Z
def linear_forward(A,W,b):

    Z=np.dot(W,A)+b

    cache=(A,W,b)

    return Z,cache
# parameters=initialize_parameters([5,4])

# Z,cache=linear_activation_forward(np.array([1,2,3,4,5]).reshape(-1,1),parameters['W1'],parameters['b1'],'relu')

# print(str(parameters['W1']))

# print(np.array([1,2,3,4,5]).reshape(-1,1))

# print(str(parameters['b1']))

# print(str(Z))

# print("##################Sigmoid##################")

# Z,cache=linear_activation_forward(np.array([1,2,3,4,5]).reshape(-1,1),parameters['W1'],parameters['b1'],'sigmoid')

# print(str(parameters['W1']))

# print(np.array([1,2,3,4,5]).reshape(-1,1))

# print(str(parameters['b1']))

# print(str(Z))

def linear_activation_forward(A,W,b,activation):

    Z,linear_cache=linear_forward(A,W,b)

    if activation=='relu':

        A,activation_cache=relu(Z)

    elif activation=='sigmoid':

        A,activation_cache=sigmoid(Z)

    cache=(linear_cache,activation_cache)

    return A,cache

        
def L_model_forward(X,parameters):

    L=len(parameters)//2

    A=X

    caches=[]

    for l in range(1,L):

        A_prev=A

        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')

        caches.append(cache)

    A_prev=A

    AL,cache=linear_activation_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')

    caches.append(cache)

    return AL,caches
# X=np.array([1,2,3,4,5]).reshape(-1,1)

# parameters=initialize_parameters([5,4,3])

# AL, caches = L_model_forward(X, parameters)

# print("AL = " + str(AL))

# print("Length of caches list = " + str(len(caches)))
def compute_cost(AL,Y):

    m=Y.shape[1]

    cost=-(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T))/m

    cost=np.squeeze(cost)

    return cost

    
def relu_backward(dA,activation_cache):

    Z=activation_cache

    return dA*np.where(Z>0,1,0)

    

def sigmoid_backward(dA,activation_cache):

    Z=activation_cache

    z,cache=sigmoid(Z)

    return dA*(z*(1-z))

def linear_backward(dZ,cache):

    A_prev,W,b=cache

    m=A_prev.shape[1]

    

    dW=np.dot(dZ,A_prev.T)/m

    db=np.sum(dZ,axis=1,keepdims=True)/m

    dA_prev=np.dot(W.T,dZ)

    return dA_prev,dW,db
def linear_activation_backward(dA,cache,activation):

    linear_cache,activation_cache=cache

    if activation=='relu':

        dZ=relu_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    elif activation=='sigmoid':

        dZ=sigmoid_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    return dA_prev,dW,db
def L_model_backward(AL,Y,caches):

    grads={}

    L=len(caches)

    m=Y.shape[1]

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache=caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):

        current_cache=caches[l]

        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)]=linear_activation_backward(dAL, current_cache, 'relu')

    return grads
def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients, output of L_model_backward

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

                  parameters["W" + str(l)] = ... 

                  parameters["b" + str(l)] = ...

    """

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]

        parameters["b" + str(l+1)] = parameters["b"+ str(l+1)]-learning_rate*grads["db"+ str(l+1)]

    return parameters
def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):#lr was 0.009

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    np.random.seed(1)

    costs = []                         # keep track of cost

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)       

        grads = L_model_backward(AL, Y, caches)        

        parameters = update_parameters(parameters, grads, learning_rate)        

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
def predict(parameters,test_X):

    L=len(parameters)//2

    A=test_X

    for l in range(1,L):

        Z=np.dot(parameters['W'+str(l)],A)+parameters['b'+str(l)]

        A=relu(Z)

    Z=np.dot(parameters['W'+str(L)],A)+parameters['b'+str(L)]

    prediction,cache=sigmoid(Z)

#     print(prediction.shape)

    return prediction

    
def score(parameters,X,Y):

#     print(Y.shape)

    return ((100-compute_cost(predict(parameters,X),Y)))


layer_dims=(x_train.shape[0],1)  

parameters = L_layer_model(x_train, y_train, layer_dims, learning_rate =0.02,num_iterations = 10000, print_cost = True)
train_score=score(parameters,x_train,y_train)

print('Accuracy on training data = '+str(round(train_score,2)))

test_score=score(parameters,x_test,y_test)

print('Accuracy on testing data = '+str(round(test_score,2)))