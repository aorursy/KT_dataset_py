# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import math
 
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
 
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
dataframe = pd.read_csv("../input/train.csv")
testdataframe = pd.read_csv("../input/test.csv")
print(dataframe.describe())
print(testdataframe.describe())
dataframe=dataframe.reindex(np.random.permutation(dataframe.index))
print(dataframe)
sex_numeric = dataframe["Sex"].apply(lambda sex : sex == 'male')
sex_numeric = sex_numeric * 1
#X= pd.DataFrame({'PassengerId','Pclass', 'Sex' : sex_numeric, 'Age','SibSp','Parcb','Fare'})
dataframe2=pd.DataFrame()
dataframe2["PassengerId"]= dataframe["PassengerId"] / 200 #normalization
dataframe2["Sex"]= dataframe["Sex"].apply(lambda sex : sex == 'male') * 1
dataframe2["Pclass"]= dataframe["Pclass"]
dataframe2["SibSp"]= dataframe["SibSp"]
dataframe2["Parch"]= dataframe["Parch"]
dataframe2["Fare"]= dataframe["Fare"] / 10 #normalization
X_train= dataframe2
Y_train= dataframe["Survived"]

print(X_train)
print(Y_train)
testdataframe2=pd.DataFrame()
testdataframe2["PassengerId"]= testdataframe["PassengerId"] /200
testdataframe2["Sex"]= testdataframe["Sex"].apply(lambda sex : sex == 'male') * 1
testdataframe2["Pclass"]= testdataframe["Pclass"]
testdataframe2["SibSp"]= testdataframe["SibSp"]
testdataframe2["Parch"]= testdataframe["Parch"]
testdataframe2["Fare"]= testdataframe["Fare"] /10
X_test= testdataframe2

print(X_test)
print("test shape =", X_test.shape)
print("train shape =", X_train.shape)
print("label shaple =", Y_train.shape)
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test =np.array(X_test)

X_train = X_train.T
Y_train= Y_train.T
X_test = X_test.T
Y_train = Y_train.reshape(1,891)
print(X_train, Y_train)
print("test shape =", X_test.shape)
print("train shape =", X_train.shape)
print("label shaple =", Y_train.shape)
def initialisation(X, n_h):
    features = X.shape[0]
    W1 = np.random.randn(n_h, features) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.zeros((1, 1))
    parameters = {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2}
    return parameters
a= np.array([[100, 200, 300, 800, 600], [4, 5, 6, 2, 9],[0.2, 0.5, 0.8, 0.10, 0.1]])
c= np.array([[1, 1, 0, 0, 1]])
p = initialisation(a, 5)
print(p)
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A
sigmoid(a)
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache
relu(a-3)
def forwardprop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1, relucache = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1" : Z1, "A1" : A1, "Z2" : Z2, "A2": A2}
    return cache, relucache
e,f= forwardprop(a, p)
print(e,f)
def drelu(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    # mapping Z to dZ
    return dZ
j,k = relu(a-3)
print(j,k)
drelu(j,k)
def cost(cache, Y):
    y = cache ["A2"]
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(y) + (1-Y) * np.log(1-y))
    cost = np.squeeze(cost)
    return cost
g= cost(e,c)
print(g)
print(c.shape)
def backprop(X, Y, cache, relucache, cost, parameters): 
    m = Y.shape[1]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    y = cache["A2"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    dZ2 = y - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = drelu(dA1, relucache)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims= True )
    gradients = {"dW2" : dW2, "db2" : db2, "dW1": dW1, "db1": db1}
    return gradients
backprop(a, c, e, f, g, p)

def NNmodel(X_train, Y_train, learning_rate, number):
    parameters= initialisation(X_train, 5)
    for i in range (number):
        cache, relucache = forwardprop(X_train, parameters)
        costs = cost(cache, Y_train)
        gradients = backprop(X_train, Y_train, cache, relucache, costs, parameters) 
        dW2 = gradients["dW2"]
        db2 = gradients["db2"]
        dW1 = gradients["dW1"]
        db1 = gradients["db1"]
        parameters["W2"] -= learning_rate * dW2
        parameters["b2"] -= learning_rate * db2
        parameters["W1"] -= learning_rate * dW1
        parameters["b1"] -= learning_rate * db1
    return parameters
modelparameters= NNmodel(X_train, Y_train, 0.1, 1000)
def predict(X_test, parameters):
    cache, _ = forwardprop(X_test, parameters)
    Y_pred = cache["A2"]
    Y_pred = Y_pred > 0.5
    return Y_pred
#this is to test the model with a input c output, by normalise the input features, it can actually predict good classification problem.
para= NNmodel(a,c,0.1,1000)
perd = predict(a,para) * 1
pcache, _ = forwardprop(a, para) 
print(pcache["A2"])
print(perd)
print(c)
Y_prediction = predict(X_test, modelparameters)
Y_pred = Y_prediction.T * 1
Y_pred = pd.DataFrame(Y_pred)
Y_pred.shape

submission = pd.DataFrame({
        "PassengerId": testdataframe["PassengerId"],
        "Survived": Y_pred[0]
    })
submission.to_csv('submission.csv', index=False)
submission.describe()

