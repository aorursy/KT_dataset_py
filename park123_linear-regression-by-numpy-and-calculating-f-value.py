
import numpy as np
import pandas as pd
import time
#Create random variable X,y
N=10000
np.random.seed(2020)
X = np.random.normal(10,10,[N,3])
X = np.c_[np.ones(N),X] # first column is ones
#print(X)
e = np.random.normal(0,10,N)
y = 5*X[:,0] - 3*X[:,1] + 1*X[:,2] + 0.1*X[:,3]+ e
#OLS algorithm
start = time.time()
beta = np.linalg.inv(X.T@X) @ X.T @ y # @ means matrix multiplication
print(time.time()-start) # consuming time
print(beta) #beta 0 to 3
#yhat = X[:,0]*beta[0] + X[:,1]*beta[1] + X[:,2]*beta[2] + X[:,3]*beta[3]
yhat = X @ beta
SSR = sum((yhat-np.mean(y))**2)
SSE = sum((y-yhat)**2)
SST = sum((y-np.mean(y))**2)
#R-square
SSR/SST
#F-value
k=3 # the number of parameters is 3
(SSR/k)/(SSE/(N-k-1))
sigma2 = ((y-yhat).T @ (y-yhat))/N #sigma square
for i in range(4):
    print(beta[i]/((sigma2*np.linalg.inv((X.T @ X))[i,i])**(1/2)))
import statsmodels.api as sm
lm = sm.OLS(y,X)
start = time.time()
lm = lm.fit()
print(time.time()-start)
lm.summary()