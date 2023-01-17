import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ex1data1 = pd.read_csv("../input/ex1data1.txt",header=None)
ex1data1.head()
ex1data1.plot.scatter(x=0,y=1)
data = ex1data1.iloc[:,0].values
y = ex1data1.iloc[:,1].values

if len(data.shape) < 2:
    data = data.reshape(-1,1)
    
ones = np.ones(data.shape[0])
X = np.c_[data,ones]
def NormalEquation(X,y):
    transfer = X.T
    return np.dot(np.dot(np.linalg.pinv(np.dot(transfer,X)) , transfer) , y)
theta = NormalEquation(X,y)
theta
def predect(test_data,theta):
    ones = np.ones(test_data.shape[0])
    test_data = np.c_[test_data,ones]
    return np.dot(test_data,theta)

plt.scatter(data,y, alpha=.8, color='navy')
y_pred = predect(np.arange(30),theta)
plt.plot(y_pred,color='red')
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


data_Normalize = featureNormalize(data)

ones = np.ones(data.shape[0])
X = np.c_[data_Normalize,ones]
def computeCost(X, y, theta):
    return np.sum((np.dot(X,theta) -y)**2)/(2*len(y));
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters);

    theta_len = len(theta);
    
    #迭代 num_iters 次
    for num_iter in range(num_iters):
        theta = theta - (alpha/m)*np.dot(X.T,(np.dot(X,theta).reshape(-1)-y))
        J_history[num_iter] = computeCost(X, y, theta)
        
    return theta, J_history
    
alpha = 0.01
num_iters = 400
theta = np.zeros(data.shape[1]+1)

theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
print(theta)
plt.plot(J_history)
def predect(test_data,theta):
    ones = np.ones(test_data.shape[0])
    test_data = np.c_[test_data,ones]
    return np.dot(test_data,theta)

plt.scatter(data,y, alpha=.8, color='navy')
y_pred = predect(np.arange(5),theta)
plt.plot(y_pred,color='red')
from sklearn import linear_model
reg = linear_model.LinearRegression()

model = reg.fit(data,y)

