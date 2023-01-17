import numpy as np
import pandas as pd
import matplotlib as plt
import random

data_titanic = pd.read_csv("...\\train.csv")
#data_titanic.head()

data_titanic.head(10)
col_del  = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data_titanic_mod = data_titanic.drop(col_del, 1)
data_titanic_mod = data_titanic_mod.replace(['male', 'female', 'S', 'C', 'Q'], [1, 0, 1,2,3])
data_titanic_mod.fillna(0, inplace=True)
data_titanic_mod1 = data_titanic_mod[data_titanic_mod.Age != 0]
Y = data_titanic_mod1['Survived']
col_del1 = ['Survived']
data_titanic_mod2 = data_titanic_mod1.replace(col_del1, 1)
data_titanic_mod2 = (data_titanic_mod1 - np.min(data_titanic_mod1))/(np.max(data_titanic_mod1) - np.min(data_titanic_mod1))
#z = np.argwhere(np.isnan(data_titanic_mod1['Age']))
#data_titanic_mod2 = data_titanic_mod1(data_titanic_mod1['Age'] != np.isnan(data_titanic_mod1) )
#data_titanic_mod2.head()
#data_titanic_mod2['const'] = np.ones(data_titanic_mod2.shape[0], dtype = int)
Input = data_titanic_mod2.copy()
#Output = Y.copy()
Y = Y[:,None]
Output = Y.copy()
Output.shape

Input.head()
#Input.shape == 713X7 
def initializtion(Input):
    random.seed(10)
    parameters1 = np.random.rand(Input.shape[1], 4)   # 7X4
    bias1 = np.random.rand(1,4)                       # 1X4
    parameters2 = np.random.rand(parameters1.shape[1], 1)  # 4X1
    bias2 = np.random.rand(1,1)                            # 1X1
    
    return parameters1, bias1, parameters2, bias2
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def forward(Input, parameter1,b1, parameter2, b2):
    
    Z1 = np.dot(Input,parameter1)+ b1    #713X4
    A1 = np.tanh(Z1)   # 713X4
    
    Z2 = np.dot(A1, parameter2) + b2     #713X1
    A2 = sigmoid(Z2)                     #713X1

    return A2, A1 
def backward(Output, Input, parameters1,bias1, parameters2, bias2):
    m = Output.shape[0]
    A2, A1 = forward(Input, parameters1,bias1, parameters2, bias2)
    dZ2 = A2 - Output     # 713X1
    dW2 = (1/m)*np.dot(np.transpose(A1), dZ2)    # 4X1
    db2 = (1/m)*np.sum(dZ2, axis = 0, keepdims = True)
    
    dA1 = np.dot(dZ2, np.transpose(parameters2))
    g1_dash = 1-np.power(A1,2)
    dZ1 = np.multiply(np.dot(dZ2, np.transpose(parameters2)), g1_dash) #713X4
    dW1 = (1/m)*np.dot(np.transpose(Input), dZ1)   
    db1 = (1/m)*np.sum(dZ1, axis = 0)
    
    
    return dW2, db2, dW1, db1
def update(parameters1, parameters2, bias1, bias2, alpha = 0.1):
    
    dW2, db2, dW1, db1 = backward(Output, Input, parameters1,bias1, parameters2, bias2)
    
    parameters1  = parameters1 - alpha*dW1
    bias1 = bias1 - alpha*db1
    
    parameters2 = parameters2 - alpha*dW2
    bias2 = bias2 - alpha*db2
    
    return parameters1, bias1,parameters2,bias2
    
def implementation(Input, Output):
    
    parameters1, bias1, parameters2, bias2 = initializtion(Input)
    
    for i in range(100):
        
        parameters1, bias1,parameters2,bias2 = update(parameters1, parameters2, bias1, bias2, alpha = 0.1)
        
    return parameters1, bias1,parameters2,bias2
parameters1, bias1,parameters2,bias2  = implementation(Input, Output)

A2, A1 = forward(Input, parameters1,bias1, parameters2, bias2)
    
A2[A2>=0.5] = 1
A2[A2<0.5] = 0
    
score = 0;
for i in range(Output.shape[0]):
    if (A2[i] - Output[i]) == 0:
        score = score +1
    else:
        score = score

m = Output.shape[0]
score = (score/m)*100    
score
  
A2
