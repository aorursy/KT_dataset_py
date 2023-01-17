# Import tensorflow and other necessary libraries

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



#file copy library

from shutil import copyfile



# copy our file into the working directory 

#copyfile(src = "../input/localfunctions/fncs (1).py", dst = "../working/fncspy.py") 

# import all of our custom functions

#from fncspy import *
# Function evaluating the sigmoid

def sigmoid(x):

    return 1/(1+np.exp(-x))


#Function evaluating the derivative of the sigmoid

def sigmoidPrime(x):

    return sigmoid(x)*(1-sigmoid(x))

def net_init(net): 

    for i in range(1,len(net['struct'])): 

        net['W'][i] = 0.01*np.random.normal(size=[net['struct'][i],net['struct'][i-1]]) 

        net['b'][i] = np.zeros([net['struct'][i],1])
def net_create(st):

	net = {'struct': st, 'W': {}, 'b': {}, 'a':{}, 'h':{} }

	net_init(net)

	return net
# Function used to evaluate the neural network 

def net_predict(net,X):

    o = np.ones([1,X.shape[0]])

    

    net['h'][0] = np.transpose(X)

    for k in range(0,len(net['W'])):

        net['a'][k+1] = np.matmul(net['b'][k+1],o) + np.matmul(net['W'][k+1],net['h'][k])

        net['h'][k+1] = sigmoid(net['a'][k+1])

        

    return np.transpose(net['h'][len(net['W'])])
def net_loss(y,yhat):

    y = y.reshape([len(y),1])

    return np.sum(-(1-y)*np.log(1-yhat)-y*np.log(yhat))
def net_missed(y,yhat):

    y = y.reshape([len(y),1])

    return np.sum(np.abs(y-(yhat>0.5)))

def net_backprop(net,X,y):

    # Performing forward propagation

    yhat = net_predict(net,X)

    

    # Initializing gradients

    nabla_b = {}

    nabla_W = {}



	# Implementation of gradients based on backpropagation algorithm

    G = yhat-y.reshape([len(y),1])

    for k in range(len(net['W']),0,-1):

        # TODO: Implement back propagation here. Make sure that your imple-

		# mentation aggregates (by adding) the contributions from all the 

		# training samples. You can infere the expected shape of the list of

		# gradients from how it is used in the 'net_train' function.

        nabla_b[k] = np.sum(G, axis = 0)

        nabla_W[k] = np.matmul(net['h'][k-1],G)

        G = np.dot(np.multiply(np.transpose(sigmoidPrime(net['a'][k])), G),net['W'][k])

    return nabla_b, nabla_W
def net_train(net,X_train,y_train,X_val,y_val,epsilon,NIter):

	# Initializing arrays holding the history of loss and missed values

    Loss = np.zeros(NIter)

    Loss_val = np.zeros(NIter)

    missed_val = np.zeros(NIter)

	

	# Simple implementation of gradient descent

    for n in range(0,NIter):

		# Computing gradient and updating parameters

        nabla_b, nabla_W = net_backprop(net,X_train,y_train)

        for k in range(0,len(net['W'])):

            net['b'][k+1] = net['b'][k+1] - epsilon*nabla_b[k+1].reshape(net['b'][k+1].shape)

            net['W'][k+1] = net['W'][k+1] - epsilon*np.transpose(nabla_W[k+1])

			

		# Computing losses and missed values for the validation set

        Loss[n] = net_loss(y_train,np.transpose(net['h'][len(net['W'])]))

        yhat_val = net_predict(net,X_val)

        Loss_val[n] = net_loss(y_val,yhat_val)

        missed_val[n] = net_missed(y_val,yhat_val)

		

		# Displaying results for the current epoch

#         print("... Epoch {:3d} | Loss_Train: {:.2E} | Loss_Val: {:.2E} | Acc_Val: {:2.2f}".format(n,Loss[n],Loss_val[n],100-100*(missed_val[n])/len(yhat_val)))

	

    return Loss, Loss_val, missed_val
# Import MNIST data

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784',version=1)



# Getting training and testing data:

# We are setting up just a simple binary classification problem in which we aim to

# properly classify the number 2.

X, y_str = mnist["data"], mnist["target"]

y = np.array([int(int(i)==2) for i in y_str])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Creating a neural network structure

net = net_create([784,100,1])



# Training neural network:

# Note that since I am not doing any hyper-parameter tuning, I am using the test sets for

# validation to show how the generalization error changes as the network gets trained. 

Loss,Loss_val,mae_val = net_train(net,X_train,y_train,X_test,y_test,epsilon=1e-6,NIter=300)
#### Plotting learning curves:

# Note that we don't observe overfitting here because the model is very simple.

plt.plot(Loss/np.max(Loss))

plt.plot(Loss_val/np.max(Loss_val))

plt.legend({'Normalized Training Loss','Normalized Validation Loss'})

plt.show()