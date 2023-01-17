# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

X = 2 * np.random.rand(100, 1)
y = 1 + 5 * X + np.random.randn(100, 1)
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
# Linlag is the linera algebra Library
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
def plot_Lerning_rate(eta,ite):
    n_iterations = 1000
    m = 100
    acc=np.empty(1000)
    nc=np.arange(1,1001,1)
    i=0
    theta = np.random.randn(2,1)  # random initialization

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        acc[i]=theta[1]
        i = i + 1
    x=pd.Series(acc[1:ite],index=nc[1:ite])
    x.plot()
    # Add title and axis names
    plt.title('Learning Growth Eta ' + str(eta))
    plt.xlabel('Iteration')
    plt.ylabel('Slope')
    plt.show() 
plot_Lerning_rate(0.001,100)
plot_Lerning_rate(0.01,100)
plot_Lerning_rate(0.1,100)
plot_Lerning_rate(0.5,100)
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    
            y_predict = X_new_b.dot(theta)           
            style = "b-" if i > 0 else "r--"         
            plt.plot(X_new, y_predict, style)        
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                

plt.plot(X, y, "b.")                                
plt.xlabel("$x_1$", fontsize=18)                    
plt.ylabel("$y$", rotation=0, fontsize=18)          
plt.axis([0, 2, 0, 15])                              
plt.show()                                     
%matplotlib inline

np.random.seed(12)
num_observations = 500

# Creating Variables for two classes

# Takes mean and covariance
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .6],[.6, 1]], num_observations)


simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
    return weights
weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 200000, learning_rate = 0.01, add_intercept=True)
print(weights)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 0.01)
clf.fit(simulated_separableish_features, simulated_labels)

print(clf.intercept_, clf.coef_)