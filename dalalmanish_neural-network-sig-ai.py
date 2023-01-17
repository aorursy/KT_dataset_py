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
import numpy as np
#Input array
X=np.array([[1,2],[1,3],[1,4],[5,8],[5,7],[5,6]])
#Output
y=np.array([[1],[1],[1],[0],[0],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
print(sigmoid(1.76))

print(wh)
print(bh)
print(wout)
print(bout)
for i in range(epoch):
#Forward Propogation
    
    hiddenlayer=sigmoid(np.dot(X,wh)+bh)
    #print("hidden_layer:")
    #print(hiddenlayer)
    
    output = sigmoid(np.dot(hiddenlayer,wout)+bout)
    #print("output:")
    #print(output)
    #error
    #print("errors:")
       
    E = y-output
   # print(E)
    slope_output_layer = derivatives_sigmoid(output)
    #print(slope_output_layer)
    d_output = E * slope_output_layer
    #print(d_output)
    der_output= hiddenlayer.T.dot(d_output)
    #print(der_output)
    
    
    Error_at_hidden_layer = d_output.dot(wout.T)
    #print( Error_at_hidden_layer )
    
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer)
    #print(slope_hidden_layer)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    #print(d_hiddenlayer)
    der_hiddenlayer=X.T.dot(d_output)
    #print(der_hiddenlayer)
    
    
   
    
    
    #Backpropagation
    wout -=  der_output*lr
    bout -= np.sum(d_output, axis=0,keepdims=True) *lr
    wh -= der_hiddenlayer *lr
    bh -= np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
   
print (output)




