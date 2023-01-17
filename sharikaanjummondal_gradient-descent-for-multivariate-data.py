import numpy as np

import pandas as pd
class LRGD:

    def __init__(self):

        self.theta=0

        self.cost=0

        

    #hypothesis function

    def hypothesis(self,X, n):  #n is the no. of features

        h = np.ones((X.shape[0],1))

        self.theta = theta.reshape(1,n+1)

        for i in range(0,X.shape[0]):

            h[i] = float(np.matmul(self.theta, X[i]))

        h = h.reshape(X.shape[0])

        return h

    

    #batch gradient descent

    def BGD(self, alpha, num_iters, h, X, y, n):    #alpha is the learning rate and num_iters is the no. of iterations.

        self.cost = np.ones(num_iters)

        for i in range(0,num_iters):

            self.theta[0] = self.theta[0] - (alpha/X.shape[0]) * sum(h - y)

            for j in range(1,n+1):

                self.theta[j] = self.theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])

            h = hypothesis(X, n)

            self.cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))

        self.theta = self.theta.reshape(1,n+1)

        return self



    #the main function in which we will use the above created function

    def fit(self,X, y, alpha=0.0001, num_iters=10000):

        n = X.shape[1]

        one_column = np.ones((X.shape[0],1))

        X = np.concatenate((one_column, X), axis = 1)

        # the parameter vector

        self.theta = np.zeros(n+1)

        # hypothesis calculation

        h = hypothesis(self, X, n)

        # returning the optimized parameters by Gradient Descent

        self.theta, self.cost = BGD(self,alpha,num_iters,h,X,y,n)

        return self

    

    #predictions

    def predit(self,X):

        X = np.concatenate((np.ones((X.shape[0],1)), X),axis = 1)

        predictions = hypothesis(self, X, X.shape[1] - 1)

        return predictions