import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
ex7data1 = pd.read_csv("../input/ex7data1.csv",header=None)
ex7data1.plot.scatter(x=0,y=1)
ex7data1.plot.scatter(x=0,y=1)
def calcSigma(X,Classes):
    mean = []
    covariance = []
    for i in range(Classes):
        u = np.mean(X[y == i],axis = 0)
        mean.append(u)
        X_u = X-u
        sig = np.dot(X_u.T,X_u)
        covariance.append(sig)
        
    np.linalg.svd(a,full_matrices=1,compute_uv=1)
    return mean,covariance
X = ex7data1.values
mean = np.mean(X,axis = 0)
X_u = X-mean
covariance = np.dot(X.T,X)
U,Sigma,V = np.linalg.svd(covariance,full_matrices=1,compute_uv=1)
print(U)
print(Sigma)
print(V)
w = U[0]
plt.plot(X_u[:, 0], X_u[:, 1],'o')
w_x = np.arange(-3,3)
w_y = np.arange(-3,3)/w[0]*w[1]
plt.plot(w_x, w_y)
def predect(X,w):
    shadows = []
    for i in range(len(X)):
        shadows.append(np.dot(w,X[i]))
    return np.array(shadows)


shadows = predect(X_u,w)
plt.plot(shadows,np.zeros(len(X_u)),'x')