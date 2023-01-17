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

y = np.array([[1,0,1,0,1],[0,1,0,1,0]]).transpose()

num_features = 2
x = np.arange(1,num_features * y.shape[0] + 1).reshape((num_features,y.shape[0])).transpose() / 10

plt.figure()
plt.plot(x[np.where(y[:,0] == 1)[0],0], x[np.where(y[:,0] == 1)[0],1], 'b+')
plt.plot(x[np.where(y[:,0] == 0)[0],0], x[np.where(y[:,0] == 0)[0],1], 'ro')
plt.title('θ1 Vs θ2');

def spline(x):
    
    x1 = (x[:,0] - x[:,1]).reshape((1,x.shape[0])).transpose()
    x2 = (x[:,0] - np.power(x[:,1],2)).reshape((1,x.shape[0])).transpose()
    x3 = (x[:,0] - np.power(x[:,1],3)).reshape((1,x.shape[0])).transpose()
    
    return np.concatenate((x1,x2,x3), axis=1)

s = lambda x: (x-np.mean(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))
g = lambda z: 1 / (1 + np.exp(-z))
h = lambda x, theta: g(x @ theta);
p = lambda x, theta: np.argmax(g(Xs @ theta), axis=1)

def gradient_descent_regularization(Xs,y,h,lp,alpha,num_iter):

    num_k = y.shape[1]
    num_features = Xs.shape[1]
    theta = np.zeros((num_features,num_k))
    J_history = np.zeros((num_iter,num_k))
    for k in range(num_k):

        k_y = y[:,k].reshape(1,len(y)).transpose()
        k_theta = np.zeros((num_features,1))
        for i in range(num_iter):
            
            J_reg = lp*(k_theta[1:].transpose() @ k_theta[1:])/(2*len(k_y))
            J = np.sum((-k_y*np.log(h(Xs,k_theta)) - (1-k_y)*np.log(1-h(Xs,k_theta)))/len(k_y)) + J_reg

            grad_reg = lp*k_theta/len(y)
            grad_reg[0] = 0
            grad = (Xs.transpose() @ (h(Xs,k_theta)-k_y))/len(k_y) + grad_reg

            k_theta = k_theta - alpha*grad

            J_history[i,k] = J

        theta[:,k] = k_theta[:,0]
        
    return (theta, J_history)

lp = 1e-7
alpha = 7.1
num_iter = 5000

print('Spline: y = θ0 + θ1 (x1 - x2) + θ2 (x1 - x2^2) + θ3 (x1 - x2^3)')
Xm = spline(x)
print('X model')
print(Xm.shape)
print(Xm)

Xs = np.concatenate((np.ones((y.shape[0],1)), s(Xm)), axis=1)
    
theta, J_history = gradient_descent_regularization(Xs,y,h,lp,alpha,num_iter)

print('Predictions')
print(p(Xs,theta))
print('Expected')
print(y[:,1])
print('Success rate %')
print(np.mean(y[:,1] == p(Xs,theta)) * 100)

plt.figure()
for k in range(y.shape[1]):
    plt.plot(J_history[:,k])
plt.title('J(θ)');

plt.figure()
plt.plot(x[np.where(y[:,0] == 1)[0],0], y[np.where(y[:,0] == 1)[0]], 'b+')
plt.plot(x[np.where(y[:,0] == 0)[0],0], y[np.where(y[:,0] == 0)[0]], 'ro')
plt.plot(x[:,0], h(Xs,theta[:,0]), 'b-')
plt.plot(x[:,0], h(Xs,theta[:,1]), 'r-')
plt.title('Training data y = f(θ1)');
