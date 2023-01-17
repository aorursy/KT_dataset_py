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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import cv2
import sys
import random
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
Train = []
Train_Samples = []
Train_Labels = []
train_path = '../input/train'
test_path = '../input/test'

dog_count = 0
cat_count = 0
Train = [img for img in os.listdir(train_path)]

print (Train[0:5])
### Generate ~ 2000 pics for dogs and cats | Cats = 0 & Dogs = 1 for numeric representation

training_set_per_example = 1000

for i,pic in enumerate(Train):
    if Train[i][0:3].lower() == 'cat' and cat_count < training_set_per_example:
        Train_Samples.append(cv2.resize(cv2.imread(train_path + '/' + Train[i]),(256,256)).flatten())    ### Cats are denoted as Zeros
        Train_Labels.append(0)
        cat_count = cat_count + 1 
        # print("Cat Image number " , pic , " added as sample number " , i , " to the training set ")
    elif Train[i][0:3].lower() == 'dog' and dog_count < training_set_per_example:
        Train_Samples.append(cv2.resize(cv2.imread(train_path + '/' + Train[i]),(256,256)).flatten())   ### Dogs are denoted as 1s
        Train_Labels.append(1)
        dog_count = dog_count + 1 
#Split Data into Train & Dev
train_x, dev_x, train_y, dev_y = train_test_split(Train_Samples,Train_Labels,train_size = 0.8, random_state = 10)
## Convert Lists to arrays to make calculations for Neural Network easy 
X_Train_temp = np.array(train_x)
Y_Train_temp = np.array(train_y)
Dev_X_temp = np.array(dev_x)
Dev_Y_temp = np.array(dev_y)

## Reshape all arrays to ensure all the dimensions are correct
X_Train = X_Train_temp.reshape(X_Train_temp.shape[1],X_Train_temp.shape[0]) / 255
Dev_X = Dev_X_temp.reshape(Dev_X_temp.shape[1],Dev_X_temp.shape[0]) / 255
Y_Train = Y_Train_temp.reshape(Y_Train_temp.shape[0],1) 
Dev_Y = Dev_Y_temp.reshape(Dev_Y_temp.shape[0],1)

Dog_Images = Y_Train[Y_Train == 1]
Cat_Images = Y_Train[Y_Train == 0]
## Print all array shapes to cross verify their size and shape 
print("Training Input Shape:" ,X_Train.shape)
print("Training Output Shape:" , Y_Train.shape)
print("Validation Input shape:" ,Dev_X.shape)
print("Validation Output Shape:" , Dev_Y.shape)
print(" ")
print("No. of Training examples is ", X_Train.shape[1])
print("No. of validation examples is ", Dev_X.shape[1])
# Here, we need to initialize W and b for the number of layers that are passed as parameter 
# Dimension of W(l) = [n(l), n(l-1)]
# Dimension of b(l) = [n(l),1]
# where n(l) is the number of neurons in lth layer and n(l-1) is the number of neurons in (l-1)th layer - this information resides in layers_dims array
# For l=1, n(l-1) will be the shape of individual input training examples i.e the shape of individual flattened image

def initialize_parameters(layers_dims,seed):
    np.random.seed(seed)
    parameters = {}
    no_of_layers = len(layers_dims) - 1 
    for l in range(1,(no_of_layers+1)):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1), dtype=np.float64)
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters
# This block will output tr output that will be passed to an activation layer
def linear_forward(X,W,b):
    Z = np.dot(W,X) + b
    
    assert(Z.shape == (W.shape[0], X.shape[1]))
    
    return Z
# This block will convert linear_forward output into a non-linear activation output using various activation functions
# This block writes code for 2 activations - "sigmoid' and 'leaky relu' with alpha value of 0.01 
def non_linear_forward(Z,activation):
    if activation == 'sigmoid':
        A = 1 /(1 + np.exp(-Z))
        
    elif activation == 'relu':
        t = np.ones(Z.shape)
        t[Z<=0] = 0.01
        A = Z * t
        
    return A
    
# This block will write code for forward propagation and store W,b and Z values corresponding to every layer in a cache

def forward_prop(X_Train,parameters):
    saved = {}
    caches = []
    Z = []
    A_prev = X_Train
    for l in range(1,(no_of_layers+1)):
        Z = linear_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)])
        A = non_linear_forward(Z,activations[l-1])
        saved['Z'+str(l)] = Z 
        cache = A_prev,parameters['W'+str(l)],parameters['b'+str(l)]
        caches.append(cache)
        A_prev = A 
    # Save last output of activation to initiate backpropagation logic
    AL = A 
    return AL,saved,caches
# Compute cost 
def compute_cost(AL,Y_Train):
    m = Y_Train.shape[1]
    cost = (-1/m) * (np.sum(np.multiply(Y_Train,np.log(AL)) + np.multiply((1-Y_Train),np.log((1-AL)))))
    cost = np.squeeze(cost)
    
    return cost
# Initialize backpropagation using AL 
def initialize_backprop(AL,Y_Train):
    dAL = - (np.divide(Y_Train, AL) - np.divide(1 - Y_Train, 1 - AL)) 
    return dAL
# Backward propagation for Activation function - Returns gradient on activation functions
def backward_prop_activation(dA,saved,activation,layer):
    if activation == 'sigmoid':
        Z = saved['Z'+str(layer)]
        s = 1/(1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        
    elif activation == 'relu':
        Z = saved['Z'+str(layer)]
        dZ = np.array(Z,copy=True)
        dZ[Z>0] = 1
        dZ[Z<0] = 0.01
        dZ = dA*dZ
       
    assert(dZ.shape == Z.shape)

    return dZ
# Backward propagation for the linear part(s)
def backward_prop_linear(dZ,caches,l):
    A_prev,W,b = caches[l-1]
    
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
          
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev,dW,db
# Store Gradients
def store_gradients(dA_prev,dW,db,layer):
    grads = {}
    grads["dA" + str(layer)] = dA_prev
    grads["dW" + str(layer)] = dW
    grads["db" + str(layer)] = db
    
    return grads
        
# Update Gradients 
def update_gradients(parameters,grads,learning_rate,layer):
    parameters["W"+str(layer)] = parameters["W"+str(layer)] - (learning_rate * grads["dW"+str(layer)])
    parameters["b"+str(layer)] = parameters["b"+str(layer)] - (learning_rate * grads["db"+str(layer)])
    
    return parameters
# Define Hyperparameters and Layers
## layers_dims contains information about [input neurons, hidden layer 1 neurons,hidden layer 2 neurons .... hidden layer n neurons, 
## output layer neurons]
layers_dims = [X_Train.shape[0],1000,1000,1]
activations = ['relu','relu','sigmoid']

# Build a full fledged neural network layer with forward and backward propagation logic
# Activation functions to be used are picked from "activations" array.


costs = []
no_of_layers = len(layers_dims) - 1
epochs = 200
seed = 192
learning_rate = 0.01

parameters = initialize_parameters(layers_dims,seed)

for i in range(epochs):
    AL, saved,caches = forward_prop(X_Train,parameters)
    cost = compute_cost(AL,Y_Train)
    costs = np.append(costs,cost)    
   
    print("Cost at epoch ", i , " is " , cost)
    Y_Train = Y_Train.reshape(AL.shape)
    dAL = initialize_backprop(AL,Y_Train)
    
    dA = dAL 

    for l in reversed(range(1,(no_of_layers+1))):
        dZ = backward_prop_activation(dA,saved,activations[l-1],l)
        dA_prev,dW,db = backward_prop_linear(dZ,caches,l)
        grads = store_gradients(dA_prev,dW,db,l)
        parameters = update_gradients(parameters,grads,learning_rate,l)
        dA = dA_prev
    
%matplotlib inline
plt.plot(costs)

print(AL)
temp = []
for x in AL.flatten():
    if x >=0.5:
        temp.append(1)
    else:
        temp.append(0)
        
cf = metrics.confusion_matrix(Y_Train.flatten(),temp)
print(cf)

acc = metrics.accuracy_score(Y_Train.flatten(),temp)
acc
metrics.f1_score(Y_Train.flatten(),temp)
# Let's randomly check mislabeled images
misclass = []
correct = []
for x,predictions in enumerate(temp):
    if temp[x] == Y_Train.flatten()[x]:
        correct.append(x)
    else:
        misclass.append(x)

print("Misclassified indexes :")
print(misclass[0:50])

print("Correct indexes :")
print(correct[0:50])
%matplotlib inline
idx = 55
print("Model predicted CAT for this image " if temp[idx] == 0 else "Model predicted DOG for this image ")
plt.imshow(X_Train_temp[idx].reshape(256,256,3))
plt.show()
# Let's Check CV Accuracy

AL_D,saved_D,caches_D = forward_prop(Dev_X,parameters)

temp_D = []
for x in AL_D.flatten():
    if x >=0.5:
        temp_D.append(1)
    else:
        temp_D.append(0)
        
cf_D = metrics.confusion_matrix(Dev_Y.flatten(),temp_D)
print(cf_D)
acc_D = metrics.accuracy_score(Dev_Y.flatten(),temp_D)
print(acc_D)
metrics.f1_score(Dev_Y.flatten(),temp_D)
