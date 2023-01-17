# necessary import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import os
# print(os.listdir("../input"))
# create a data set
np.random.seed(42)
X = np.random.random((4, 2))
X = np.round_(X, 2)
y = np.array([1, -1, 1, 1])
print(X)
print(y)
# generate initial value of estimator 
rgen = np.random.RandomState(42)
w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
print(w)
# evaluate the net input 
xw = np.dot(X[0], w[1:])
xw_intercept = xw + w[0]
print(xw)
print(xw_intercept)
# Pridiction from a single iteration
y_hat = np.where(xw_intercept >= 0.0, 1, -1)
print(y_hat)
eta = 0.01
target = y[0]
update = eta * (y[0] -y_hat)
print(update)
w[1:] += update * X[0]
print('With out Intercept:', w)
w[0] += update
print('With Intercept:', w)
errors_ = [] 
error =0
error += int(update != 0.0) # give out put 1 or 0
print(error)
errors_.append(error)
# sample data
np.random.seed(42)
X = np.random.random((4,2))
X = np.round_(X, 2)
y = np.array([1, -1, 1, 1])
print(X)
print(y)
# enitiate the value of estimator
rgen = np.random.RandomState(42)
w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
print(w)
# function to calculate the net input w_transopseX
def net_input(x):
    return np.dot(x, w[1:]) + w[0]
# create the threshlold function for prediction
def prediction(x):
    return np.where(net_input(x) >= 0.0, 1, -1)
# lets look over one single loop
result_list = []
for x_i, y_i in zip(X, y):
    dict_ = {'x': x_i,
             'y': y_i,
             'net_input': round(net_input(x_i), 3),
             'y_hat': prediction(x_i)
            }
    result_list.append(dict_)
result_list
eta = 0.01
error_list = []
errors = 0
for x_i, y_i in zip(X, y):
    update = eta * (y_i - prediction(x_i))
    w[1:] += update * x_i
    w[0] += update
    errors += int(update != 0.0)
error_list.append(errors)
error_list
# iterate the same process multiple time
eta = 0.01
error_list = []
for _ in range(10):
    errors = 0
    for x_i, y_i in zip(X, y):
        update = eta * (y_i - prediction(x_i))
        w[1:] += update * x_i
        w[0] += update
        errors += int(update != 0.0)
    error_list.append(errors)
error_list
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of 
          samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, 
                              size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_i, y_i in zip(X, y):
                update = self.eta * (y_i - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
class LogisticSimulator(object):
    """Logistic data generator with 2 feature.

    Parameters
    ------------
    size : sample size (int)
      Any value between 0 and Infinity.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    X_ : nd-array
      Feature metrics
    y : 1d-arry
      Outcome variable """

    def __init__(self, size=4000, random_state=13):
        self.size = size
        self.random_state = random_state

    def sample_generator(self):
        np.random.seed(self.random_state)
        x1_ = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], self.size)
        x2_ = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], self.size)
        X_ = np.vstack((x1_, x2_)).astype(np.float32)
        y_ = np.hstack((np.ones(self.size)*-1, np.ones(self.size)))
        return X_, y_
    
    def viz_generator(self, X, y):
        plt.figure(figsize=(12,8))
        plt.scatter(X[:, 0], X[:, 1], c = y, alpha = .3)
# call the class and generate feature metrix and outcome variable
lgen = LogisticSimulator(size=1000, random_state=13)
X, y = lgen.sample_generator()
lgen.viz_generator(X, y)
ppn = Perceptron(eta=0.01, n_iter=500)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()