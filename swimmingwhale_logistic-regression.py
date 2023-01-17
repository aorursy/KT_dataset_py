import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_rows', 10)
df = pd.read_csv("../input/ex2data1.txt",header=None)
df.head()
df.plot.scatter(x=0,y=1,c=df[2].map({0:'b', 1:'r'}))
def featureNormalize(X):
    (m,n) = X.shape
    X_norm = X
    mu = np.zeros(n);
    sigma = np.zeros(n);

    for i in range(n):
        
        mu[i] = np.mean(X[:,i])
        sigma[i] = np.std(X[:,i])

        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]
        
    return X_norm
X_data = df.iloc[:,0:2].values
y = df.iloc[:,2].values

X_data = featureNormalize(X_data)

ones = np.ones(X_data.shape[0])
X = np.c_[X_data,ones]
def logistic(X,theta):    
    linear = np.dot(X,theta)
    return 1/(1+np.exp(-linear))
def computeCost(X, y, theta):
    J = -np.dot(y.T,np.log(logistic(X,theta))) - np.dot((1-y).T,np.log(1-logistic(X,theta)))
    return J/len(y);
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters);

    theta_len = len(theta);

    #迭代 num_iters 次
    for num_iter in range(num_iters):
        theta = theta - (alpha/m)*np.dot(X.T,(logistic(X,theta).reshape(-1)-y))
        J_history[num_iter] = computeCost(X, y, theta)
        
    return theta, J_history
alpha = 0.01
num_iters = 400
theta = np.zeros(3)

theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
plt.plot(J_history)
y_pred = logistic(X,theta)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
plt.scatter(X[y_pred==0][:,0], X[y_pred==0][:,1], alpha=.8, color='navy')
plt.scatter(X[y_pred==1][:,0], X[y_pred==1][:,1],alpha=.8, color='turquoise')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(df.iloc[:,:1].values, df.iloc[:,2].values)