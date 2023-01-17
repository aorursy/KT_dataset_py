# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pp
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

#Load our Dataset for Logistic Regression
components = pd.read_csv('../input/ex2data2.txt', header=None, names = ['feature 1', 'feature 2', 'faulty'])
components.head()
# get positive and negative samples for plotting
pos = components['faulty'] == 1
neg = components['faulty'] == 0

# Visualize Data
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
axes.legend(title='Legend', loc = 'best' )
axes.set_xlim(-1,1.5)
axes.set_xlim(-1,1.5)
# define function to map higher order polynomial features
def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    
    return res
# Get the features 
X = components.iloc[:, :2]
degree = 2
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
# Get the target variable
y = components.iloc[:, 2]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def costFunc(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    term1 = y * np.log(h)
    term2 = (1- y) * np.log(1 - h)
    J = -np.sum(term1 + term2, axis = 0) / m
    return J 
# Set initial values for our parameters
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
# Now call the optimization routine
#NOTE: This automatically picks the learning rate
from scipy.optimize import minimize
res = minimize(costFunc, initial_theta, args=(X_poly, y))
# our optimizated coefficients
theta = res.x
# define a function to plot the decision boundary
def plotDecisionBoundary(theta,degree, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    X_poly = mapFeature(U, V, degree)
    Z = X_poly.dot(theta)
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
    axes.legend(labels=['good', 'faulty', 'Decision Boundary'])
    return cs
# Plot Decision boundary
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes)
# set degree = 1
degree = 1
# map features to the degree
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
# set initial parameters
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
# Run the optimzation function
res = minimize(costFunc, initial_theta, args=(X_poly, y))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes)
# set degree = 6
degree = 6
# map features to the degree
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
# set initial parameters
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
# Run the optimzation function
res = minimize(costFunc, initial_theta, args=(X_poly, y))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes)
# define the regularized cost function
def costFuncReg(theta, X, y, reg_factor):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    term1 = y * np.log(h)
    term2 = (1- y) * np.log(1 - h)
    J = -np.sum(term1 + term2, axis = 0) / m
    
    # Regularization Term
    reg_term = (reg_factor * sum(theta[1:] ** 2)) / (2 * m)
    J = J + reg_term
    return J  
# Set the regularization factor to 1
reg_factor = 1

# set degree = 6
degree = 6
# map features to the degree
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
# set initial parameters
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
# Run the optimzation function with regularization factor passed to the cost function
res = minimize(costFuncReg, initial_theta, args=(X_poly, y, reg_factor))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes)
# set the regularization factor to 100
reg_factor = 100
# set degree = 6
degree = 6
# map features to the degree
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
# set initial parameters
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
# Run the optimzation function with regularization factor passed to the cost function
res = minimize(costFuncReg, initial_theta, args=(X_poly, y, reg_factor))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
fig, axes = pp.subplots();
axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes)
