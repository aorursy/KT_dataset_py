# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
class LinearRegression:

    

    

    def __init__(self, X, y, max_iter=1000):

        

        self.X = self.add_bias(X)

        self.y = y

        self.weights = np.random.randn(self.X.shape[1])

        self.max_iter = max_iter

        

    def cost(self):

        

        return sum((self.predict(self.X) - self.y)**2)/(2* len(self.y))

    

    def gradient(self, weights):

        

        return np.array([(1/self.X.shape[0]) * sum((self.predict(self.X) - self.y) * (self.X[:, w])) for w in range(len(weights))])

        

    def gradient_descent(self, learning_rate = 0.01, loop=0):

        

        while loop < self.max_iter:

            

            update = learning_rate * self.gradient(self.weights)

            self.weights = self.weights - update

            loop+=1

          

        return self

        

        

    def add_bias(self, X):

        

        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        

    def predict(self, X):

      

        return np.array([np.dot(self.weights, x) for x in X])
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
feats = train.dtypes[train.dtypes != "object"].index[1:]

for feat in feats[:-1]:

    test[feat] = test[feat].fillna(test[feat].mean())

    train[feat] = train[feat].fillna(train[feat].mean())



targets = train['SalePrice'].as_matrix()

train_features = StandardScaler().fit_transform(train[feats[:-1]].as_matrix())

test_features = StandardScaler().fit_transform(test[feats[:-1]].as_matrix())

selector = SelectKBest(f_regression, 20)

selector.fit(train_features, targets)

train_features = selector.transform(train_features)

test_features = selector.transform(test_features)
test_features.shape
reg = LinearRegression(train_features, targets, 10000)

reg.gradient_descent()



pd.DataFrame({

    'Id':test['Id'],

     'SalePrice':reg.predict(reg.add_bias(test_features))

}).to_csv('sub.csv',index=False)
plt.scatter(targets, reg.predict(reg.X))

plt.title("Actual vs Predicted")

plt.show()