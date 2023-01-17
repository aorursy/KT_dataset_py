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
import pandas as pd
import matplotlib.pyplot as plt
import os

# change directory to the location containing the data for the notebook
os.chdir(r'/kaggle/input/coursera-andrewng-ml-dataset')
os.getcwd()
data = pd.read_csv('ex2data2.txt', names=['Test1', 'Test2', 'Status'])
# input feature matrix containing two column vectors representing the score of the two tests
X = data.iloc[:, 0:2].values
# output variable vector representing whether the chip was accepted or rejected. Note this is a numpy 1d array
y1d = data.iloc[:, 2].values
data.head()
def plot_raw_data(X, y):
    accepted = y == 1
    X_accepted = X[accepted]
    X_rejected = X[~accepted]
    fig, ax = plt.subplots(figsize=(10,6)) 
    plt.plot(X_accepted[:, 0], X_accepted[:, 1], '+', ms=8, mew=2, color='black', label='accepted')
    plt.plot(X_rejected[:, 0], X_rejected[:, 1], 'o', ms=8, mew=1, mfc='cyan', mec='k',label='rejected')
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    return fig, ax
    
fig, ax = plot_raw_data(X, y1d)    
plt.title('Scatter plot of training data')
plt.legend()
plt.show()
def get_degree_pairs(degree):
    degree1 = 0
    degree2 = degree
    list_degree_pairs = []
    while degree1 <= degree and degree2 >= 0:
        if (degree1 + degree2) == degree:
            list_degree_pairs.append((degree1, degree2))
            degree1 += 1
            degree2 -= 1
    return list_degree_pairs

def feature_mapping(x1, x2, degree):    
    ones = np.ones(len(x1)).reshape(-1, 1)    
    out_X = np.concatenate((ones, x1.reshape(-1, 1)), axis=1)
    out_X = np.concatenate((out_X, x2.reshape(-1, 1)), axis=1)
    degree_counter = 2
    while degree_counter <= degree:
        list_degree_pairs = get_degree_pairs(degree_counter)
        for degree_pair in list_degree_pairs:
            # We are doing an element wise multiplication of the two vectors of size mx1 to create polynomial features
            temp = np.multiply(x1**degree_pair[0], x2**degree_pair[1])            
            out_X = np.concatenate((out_X, temp.reshape(-1, 1)), axis=1)
        degree_counter += 1
    return out_X
# test 1 scores
x1 = X[:, 0]
# test 2 scores
x2 = X[:, 1]
X_poly = feature_mapping(x1, x2, 6)
# import the sigmoid function available in scipy
from scipy.special import expit

def cost_function_reg(theta, X, y, lamda):
    theta = theta.reshape(-1, 1)
    h = expit(np.dot(X, theta))
    m = len(X)    
    # J is a scalar 
    J = -(1/m) * (np.dot(np.transpose(y), np.log(h)) + np.dot(np.transpose(1 - y), np.log(1 - h))) + (lamda/(2*m)) * np.sum(theta[1:]**2)
    grad = (1 / m) * np.dot(np.transpose(X), (h - y))   
    #print(grad.shape)
    #print(grad)
    grad_reg = (lamda / m) * theta[1:]    
    #print(grad_reg.shape)
    grad_reg = np.insert(grad_reg, 0, 0)    
    #print(grad_reg.shape)
    grad = grad + grad_reg.reshape(-1, 1)
    return J, grad
n = X_poly.shape[1]
initial_theta = np.zeros(n)
y = y1d.reshape(-1, 1)
cost, grad = cost_function_reg(initial_theta, X_poly, y, 1)
print(cost[0][0])
print('Gradient at initial theta (zeros) - first five values only:')
# format. * unpacks the arguments so if there are, say 4 placeholders and 4 elements in your list, then format unpacks the args and fills the slots.
print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(*grad.flatten()[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')
from scipy.optimize import minimize

def get_optimized_cost_theta(lamda):
    options= {'maxiter': 100}
    opt = minimize(fun=cost_function_reg, x0=initial_theta, args=(X_poly, y, lamda), jac=True, method='TNC', options=options)
    optimized_theta = opt.x
    cost_optimized = opt.fun
    return cost_optimized, optimized_theta    

opt_cost1, opt_theta1 = get_optimized_cost_theta(1.00)    
print('Optimized cost: {}'.format(opt_cost1))
print('Optimized theta: {}'.format(opt_theta1))
def predict(theta, X):
    product = np.dot(X, theta)
    classifier = lambda item: 1 if item > 0 else 0
    v_classifier = np.vectorize(classifier)
    return v_classifier(product)

predicted_y = predict(opt_theta1, X_poly)
is_correct = predicted_y == y1d
accuracy = len(predicted_y[is_correct]) / len(is_correct)
print('Accuracy of predictions: {}'.format(round(accuracy, 2)))
def plot_decision_boundary(theta, X, y):
    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label='decision boundary')
        plt.legend()
        plt.show()
    else:
        # Here is the grid range
        x1 = np.linspace(-1, 1.5, 50);
        #print(x1)
        x2 = np.linspace(-1, 1.5, 50);
        #print(x2)
        z = np.zeros((len(x1), len(x2)));
        # Evaluate z = theta*x over the grid
        for i in range(1, len(x1)):
            for j in range(1, len(x2)):
                z[i,j] = np.dot(feature_mapping(np.array([x1[i]]), np.array([x2[j]]), 6), theta)        
        z = np.transpose(z)
        plt.contour(x1, x2, z, levels=[0], linewidths=2, colors='g')
        #plt.contourf(x1, x2, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

plot_raw_data(X, y1d)  
plot_decision_boundary(opt_theta1, X_poly, y)
plt.title('Decision Boundary with lambda = 1')
plt.legend()
plt.show()
opt_cost100, opt_theta100 = get_optimized_cost_theta(0.00)    
plot_raw_data(X, y1d)
plot_decision_boundary(opt_theta100, X_poly, y)
plt.title('Decision Boundary with lambda = 0 (Overfitting)')
plt.legend()
plt.show()
opt_cost100, opt_theta100 = get_optimized_cost_theta(100.00)    
plot_raw_data(X, y1d)
plot_decision_boundary(opt_theta100, X_poly, y)
plt.title('Decision Boundary with lambda = 0 (Underfitting)')
plt.legend()
plt.show()
