# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base = pd.read_csv('../input/my-data/data04.csv', names=['x1','x2','y'], sep=';')
base.head()
class Perceptron():
    # Initialization function
    def __init__(self, learning_rate=0.01, num_iters=1000):
        self.lr = learning_rate
        self.num_iters = num_iters
        self.activation_func = self._unit_step_func_
        self.weights = None
        self.bias = None
        
    def _unit_step_func_(self, x):
        # Because we want to work not only with one-dimensional x, but also with a vector x:
        return np.where(x>=0, 1, 0)
    
    def _train_fit_(self, X_train, Y_train):
        # Let's get the dimensions of the X_train vector
        # (the number of rows is the number of samples, the number of columns is the number of features):
        num_samples, num_features = X_train.shape
        # Initializing weight (set to zero at the beginning) for each feature:
        self.weights = np.zeros(num_features)
        # Initializing bias:
        self.bias = 0
        # Now we want to make sure that our Y_train only consists of classes 0 and 1
        # and convert all the values to 0 or 1 if is not already the case:
        Y_ = np.array([1 if i>0 else 0 for i in Y_train])
        
        # Perceptron update rule:
        # w = w + delta(w), where delta(w) = a*(Y_train_i - Y_train_pred_i)*X_train_i, a - learning rate in [0,1]
        for _ in range(self.num_iters):
            for index, x_i in enumerate(X_train): # x_i - it's current sample and index - it's index x_i
                linear_output = np.dot(x_i, self.weights) + self.bias
                Y_train_pred_i = self.activation_func(linear_output)
                update = self.lr * (Y_[index] - Y_train_pred_i)
                self.weights += update * x_i
                self.bias += update * 1
            
    def _predict_(self, X_test):
        # Y_pred = g(f(w,bias)) = g(w^T*X_test+bias), where g - activation function
        # w^T*X_test it is the dot product, therefore: 
        linear_output = np.dot(X_test, self.weights) + self.bias
        Y_pred = self.activation_func(linear_output)
        return Y_pred
# One more function:
def accuracy(Y_true, Y_pred):
    dividend = 0; # Number of coincidences in vectors Y_true and Y_pred
    for i in range(len(Y_true)):
        if (Y_true[i] == Y_pred[i]):
            dividend += 1
    accuracy = dividend / len(Y_true)
    return accuracy
# Let's test Perceptron class:
X = base.loc[:, ('x1', 'x2')] #independent features
#Y = base.loc[:, ('y')]  #dependent features
Y = base[base.columns[2:]].to_numpy()
#Breakdown of data into 80% training and 20% test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#Convert to array:
X_train = X_train[["x1", "x2"]].to_numpy()
X_test  = X_test[["x1", "x2"]].to_numpy()
#Y_train = np.array(Y_train.tolist())
#Y_test  = np.array(Y_test.tolist())

Per = Perceptron(learning_rate=0.01, num_iters=1000)
Per._train_fit_(X_train, Y_train)
Y_pred = Per._predict_(X_test)
print('Accuracy of perceptron classification: ', accuracy(Y_test, Y_pred))
Per.weights
Per.bias
# Let's plot it:
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111) #Row, column, index
plt.scatter(X_train[:,0], X_train[:,1], marker = '*', c = Y_train)