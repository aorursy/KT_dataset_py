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
import matplotlib.pyplot as pp
%matplotlib inline
# Lets create some artificial data for this 
X = np.linspace(1,10, num=50)
np.random.seed(101)
r = r = np.random.uniform(-1, 1, (50,)) *20
y = (X ** 2) + r
df = pd.DataFrame(np.column_stack((X,y)), columns=['feature', 'target'])
df.head()
fig, ax = pp.subplots()
ax.scatter(df['feature'], df['target'], color='r', marker='x')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
def costfunc(theta, X, y):
    m = X.shape[0]
    y_pred = X.dot(theta)
    J = (((y_pred-y) ** 2).mean()) / 2
    return J
def mapFeatures(X, degree):
    X_poly = X
    for i in range(2, degree + 1):
        X_poly = np.column_stack((X_poly, X ** i))
    return X_poly
# Lets start by defining the degree 1
degree = 1
# get our feature and target variables
X = df['feature'].values
y = df['target'].values
# generate additonal features (if applicable)
X = mapFeatures(X, degree)

# Because the degree was 1, we did not generate any polynomial features as confirmed by the shape below
X.shape
# Add the intercept term
X = np.column_stack((np.ones(X.shape[0]), X))
X.shape
# Set initial values of our parameters
initial_theta = np.ones(degree + 1).reshape(degree + 1,1)
initial_theta.shape
# Now we minimize our cost function using scipy
from scipy.optimize import minimize
res = minimize(costfunc, initial_theta, args=(X,y))
# Optimized values of coefficients
theta = res.x
theta
# now create points to visualize the fit
X_pred = np.linspace(X[:,1].min(), X[:,1].max(), 10)
X_pred = np.column_stack((np.ones(X_pred.shape[0]), mapFeatures(X_pred, degree)))

# Now use our hypothesis to get values of y_pred
y_pred = X_pred.dot(theta.reshape(theta.shape[0],1))
# Visualize our fit
fig, ax = pp.subplots()
ax.scatter(df['feature'], df['target'], color='r', marker='x')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
ax.plot(X_pred[:,1], y_pred, color='blue')
# Now lets try to increase the degree gradually

# Helper function
def run_model(df, degree):
    # get our feature and target variables
    X = df['feature'].values
    y = df['target'].values
    
    # generate additonal features (if applicable)
    X = mapFeatures(X, degree)
    # Add the intercept term
    X = np.column_stack((np.ones(X.shape[0]), X))
    initial_theta = np.ones(degree + 1).reshape(degree + 1,1)
    
    res = minimize(costfunc, initial_theta, args=(X,y))
    # Optimized values of coefficients
    theta = res.x
    # now create points to visualize the fit
    X_pred = np.linspace(X[:,1].min(), X[:,1].max(), 10)
    X_pred = np.column_stack((np.ones(X_pred.shape[0]), mapFeatures(X_pred, degree)))

    # Now use our hypothesis to get values of y_pred
    y_pred = X_pred.dot(theta.reshape(theta.shape[0],1))
    # Visualize our fit
    fig, ax = pp.subplots()
    ax.scatter(df['feature'], df['target'], color='r', marker='x')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.plot(X_pred[:,1], y_pred, color='blue')
    
# degree = 4
run_model(df, 3)
run_model(df, 6)
# Regularized Cost Function
def costfuncReg(theta, X, y, reg_factor):
    m = X.shape[0]
    y_pred = X.dot(theta)
    J = (((y_pred-y) ** 2).mean()) / 2
    
    # add regularization term
    reg_term = (reg_factor * np.sum((theta ** 2)))/ (2 * m )
    J = J + reg_term
    return J
# New Helper for Regularized Cost Func
# Helper function
def run_model_reg(df, degree, reg_factor):
    # get our feature and target variables
    X = df['feature'].values
    y = df['target'].values
    
    # generate additonal features (if applicable)
    X = mapFeatures(X, degree)
    # Add the intercept term
    X = np.column_stack((np.ones(X.shape[0]), X))
    initial_theta = np.ones(degree + 1).reshape(degree + 1,1)
    
    res = minimize(costfuncReg, initial_theta, args=(X,y,reg_factor))
    # Optimized values of coefficients
    theta = res.x
    # now create points to visualize the fit
    X_pred = np.linspace(X[:,1].min(), X[:,1].max(), 10)
    X_pred = np.column_stack((np.ones(X_pred.shape[0]), mapFeatures(X_pred, degree)))

    # Now use our hypothesis to get values of y_pred
    y_pred = X_pred.dot(theta.reshape(theta.shape[0],1))
    # Visualize our fit
    fig, ax = pp.subplots()
    ax.scatter(df['feature'], df['target'], color='r', marker='x')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.plot(X_pred[:,1], y_pred, color='blue')
# lets run the model with degree = 6 and regularization factor = 3
run_model_reg(df, 6, 3)