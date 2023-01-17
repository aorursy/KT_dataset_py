# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
def initialize_parameters(train_set):
    length = train_set.shape[0]
    w = np.random.randn(1, length)
    b = 0
    return w, b
def forward_prop(w, b, X):
    z = np.dot(w, X) + b
    return z
def cost_function(z, y):
    m = y.shape[1]
    J = (1 / (2 * m)) * (np.sum((z - y) ** 2))
    return J
def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1 / m) * (z - y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw, db
def gradient_descent(w, b, dw, db, rate):
    w = w - (rate * dw)
    b = b - (rate * db)
    return w, b
def linear_regression(X_train, y_train, X_val, y_val, rate, epochs):
    
    y_train = np.array([y_train])
    y_val = np.array([y_val])
    X_train = X_train.T
    X_val = X_val.T
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]
    
    X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
    X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))
    
    w, b = initialize_parameters(X_train)
    
    rmse_train_list = []
    rmse_val_list = []

    for i in range(1 , epochs + 1):
        z_train = forward_prop(w, b, X_train)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
    
        rmse_train = (np.sum((z_train - y_train) ** 2) / m_train) ** 0.5
        rmse_train_list.append(rmse_train)
    
        z_val = forward_prop(w, b, X_val)
        cost_val = cost_function(z_val, y_val)
        rmse_val = (np.sum((z_val - y_val) ** 2) / m_val) ** 0.5
        rmse_val_list.append(rmse_val)

        w, b = gradient_descent(w, b, dw, db, rate)
        
    plt.plot(range(1, epochs + 1), rmse_train_list, c = 'blue', label = 'Train Set')
    plt.plot(range(1, epochs + 1), rmse_val_list, c = 'red', label = 'Test Set')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE of Test and Train sets')
    plt.title('Linear Regression with Learning rate ' + str(rate))
from sklearn.datasets.samples_generator import make_regression
X, y = make_regression(n_samples = 500, n_features = 10, n_informative = 5, noise = 20, bias  = 100, random_state = 1)

f = X.shape[1]

fig = plt.figure(figsize = (15, 15))

for i in range(1, f + 1):
    ax = fig.add_subplot(5, 5, i)
    ax.scatter(X[:, (i - 1)], y)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 5)
linear_regression(X_train, y_train, X_val, y_val, 0.01, 150)
def linear(X_train, y_train, X_val, y_val, rate, epochs):

    y_train = np.array([y_train])
    y_val = np.array([y_val])
    X_train = X_train.T
    X_val = X_val.T
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]
    
    X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
    X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))
    
    w, b = initialize_parameters(X_train)

    for i in range(1 , epochs + 1):
        z_train = forward_prop(w, b, X_train)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent(w, b, dw, db, rate)

    z_train = forward_prop(w, b, X_train)
    cost_train = cost_function(z_train, y_train)
    rmse_train = (np.sum((z_train - y_train) ** 2) / m_train) ** 0.5

    z_val = forward_prop(w, b, X_val)
    cost_val = cost_function(z_val, y_val)
    rmse_val = (np.sum((z_val - y_val) ** 2) / m_val) ** 0.5

    print(X_val, y_val, z_val)
    print('/n')
    print('The linear regression with learning rate ', str(rate), ' and ', str(epochs), ' epochs, produces predictions with train set RMSE as ', str(rmse_train), ' and test set RMSE as ', str(rmse_test))