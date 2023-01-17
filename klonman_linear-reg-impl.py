# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
# gen x and y

# split X and Y to train/test set

m = 100

n = 4

X = np.random.randint(low=0, high=100, size=(m, n+1))

Y = np.random.randint(low=500, high=1000, size=m)

X[:,0]=1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train.shape
# implement linear reg

class LinearRegModel(object):

    def __init__(self, alpha=0.000001, n_iter=500):

        self.W = None

        self.alpha = alpha

        self.n_iter = n_iter

        self.J_hist = []

        self.min_cost = None

        pass

    

    def fit(self, X, Y):

        m = len(Y)

        n_col = X.shape[1]

        self.W = np.zeros(n_col)



        # do gradient decent

        for it in range(self.n_iter):

            tmp_W = np.copy(self.W)

            error = X @ tmp_W - Y

            for j in range(n_col):

                gradient = np.sum( error * X[:,j] ) / m

                self.W[j] = self.W[j] - self.alpha * gradient

            cost = np.sum( ( X @ self.W - Y )**2 ) / (2*m) 

            self.J_hist.append(cost)

        plt.plot(self.J_hist)

        self.min_cost = self.J_hist[-1]

        print(self.min_cost, self.W)

    

    def predict(self, X):

        return X @ self.W

    
g = LinearRegModel()

g.fit(X_train, Y_train)

g.predict(X_test)