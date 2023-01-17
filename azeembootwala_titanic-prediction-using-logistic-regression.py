import numpy as np 

import matplotlib.pyplot as plt 

import pandas

import os 
os.listdir()

df = pandas.read_csv("../input/train_data.csv")

df.head()# Analysing the data set and its columns 
def get_data(string):

    if string=="train":

        df = pandas.read_csv("../input/train_data.csv")

       

    else:

        df=pandas.read_csv("../input/test_data.csv")

        

        

    T = np.asarray(df.ix[:,"Survived"]).reshape(df["Survived"].shape[0],1)

    X=np.asarray(df.ix[:,"Sex":])# splitting our input & target variables 

    #adding bias

    bias = np.ones((df["Survived"].shape[0],1))

    X = np .hstack([bias,X])

    return X,T

def sigmoid(Z):

    Z =np.exp(-Z)

    return 1/(1+Z)

def forward(X,W):

    out = sigmoid(X.dot(W))

    return out

def cross_entropy(T,Y):

    return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()

def classification_rate(T,Y):

    return np.mean(T==Y)
X,T = get_data("train")

X_test , Y_test = get_data("test")

N,D = X.shape

W = np.random.randn(D,1) # setting up the weight matrix 

learning_rate = 10e-3

reg =10e-1

cost=[]

#running gradient descent now 

for i in range(0,2000):

    Y_train=forward(X,W)

    W = W - learning_rate*(X.T.dot(Y_train-T)+reg*np.sign(W))#We used l1 regularization here 

    if i%20 ==0:

        Y_pred = forward(X_test,W)

        c = cross_entropy(Y_test,Y_pred)

        cost.append(c)

        r=classification_rate(Y_test,np.round(Y_pred))

        R =classification_rate(T,np.round(Y_train))



plt.plot(cost)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.show()

print("Classification rate on training set",R)

print("Classification rate on test set ",r)