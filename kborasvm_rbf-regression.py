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


from scipy import *
from scipy.linalg import norm, pinv
 
from matplotlib import pyplot as plt
from random import shuffle
from sklearn.linear_model import LogisticRegression
 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 0.09
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
         
        #print ("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print (G)
         
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
 
      
if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------
    n = 100
     
    x = mgrid[-5:5:complex(0,n)].reshape(n, 1)
    # set y and add random noise
    #y = sin(3*(x+0.5)**3 - 1)
    #testing random y mapping - kbora
    #print("first 10 input before shuffle= ", x[:10]) 
    shuffle(x)
    y = (x)**2/5;
    #y = y[:75]
    # y += random.normal(0, 0.1, y.shape)
    #print("first 10 input = ", x[:10]) 
    #print("first 10 output = ", y[:10]) 
    # rbf regression
    rbf = RBF(1, 6, 1)
    #rbf.train(x, y)
    #x = x.reshape(n,1)
    rbf.train(x[:75], y[:75])
    test_data = mgrid[-15:15:complex(0,n)].reshape(n, 1)
    #x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
    #z = rbf.test(x)
    z = rbf.test(x[75:])
       
    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x[:75], y[:75], 'ko')
    #plt.plot(x, y, 'ro')
     
    #plt.plot(x, (test_data**2)/5, 'g-')    
    # plot learned model
    plt.plot(x[75:], z, 'r^', linewidth=2)
    #plt.plot(x, z, 'bo', linewidth=2)
     
    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
     
    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='blue', linewidth=0.2)
        #print("cx: ", cx, "cy: ", cy)
     
    plt.xlim(-10, 10)
    plt.show()
