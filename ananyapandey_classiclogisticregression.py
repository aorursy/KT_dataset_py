# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

from numpy import log # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Classifier.csv')

X = np.array(data.iloc[:,0:2])

y = np.array(data.iloc[:,2])
import matplotlib.pyplot as plt

def plotData(X,y):

    pos=np.where(y==1)

    neg=np.where(y==0)

    plt.scatter(X[pos,0],X[pos,1],marker='o',c='black')

    plt.scatter(X[neg,0],X[neg,1],marker='x',c='red')

    plt.show()

plotData(X,y)
def sigmoid(X):

    return 1/(1+np.exp(-1.0*X))
def costfunction(theta,X,y):

    m=len(y)

    j = (-1/m)*np.sum( y * log(sigmoid(np.dot(X,theta))) + (1-y)*log(1-sigmoid(np.dot(X,theta))))

    return j
def feature_normalise(X):

    mu = X.mean(axis =0)

    sigma = np.std(X,axis=0,ddof=1)

    return (X - mu)/sigma
def gradient(theta,X,y):

    #grad = np.zeros(shape=theta.shape)

    hypo = sigmoid(np.dot(X,theta))

    grad = np.dot(np.transpose(X),(hypo - y))

    return grad
def predict(theta,X):

    p = np.zeros(shape=(X.shape[0],1))

    hypothesis = sigmoid(np.dot(X,theta))

    return p > 0.5
def learn_lostisticreg(X,y):

    #theta=np.zeros(shape=(X.shape[1],1))

    theta = np.random.rand(X.shape[1])

    from scipy import optimize as opt

    return opt.fmin_cg(f=costfunction,x0=theta,args=(X,y))
def insert_ones(X):

    X_bias = np.c_[np.ones(X.shape[0]),X]

    return X_bias



def map_feature(x1, x2):

    '''

    Maps the two input features to quadratic features.

    Returns a new feature array with more features, comprising of

    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...

    Inputs X1, X2 must be the same size

    '''

    x1.shape = (x1.size, 1)

    x2.shape = (x2.size, 1)

    degree = 6

    out = np.ones(shape=(x1[:, 0].size, 1))



    m, n = out.shape



    for i in range(1, degree + 1):

        for j in range(i + 1):

            r = (x1 ** (i - j)) * (x2 ** j)

            out = np.append(out, r, axis=1)



    return out
X_feature = map_feature(X[:, 0], X[:, 1])

X_feature=feature_normalise(X_feature)

X_bias = insert_ones(X_feature)

theta=np.zeros(shape=(X_bias.shape[1],1))

print ("Initial Error : ",costfunction(theta,X_bias,y))

print("Learning theta optimum using fmin")

newtheta = learn_lostisticreg(X_bias,y)

print(newtheta.shape)

print("Error after learning :",costfunction(newtheta,X_bias,y))