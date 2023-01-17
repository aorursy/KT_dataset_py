from sklearn.datasets import load_boston

import pandas as pd



boston = load_boston()



X = boston.data

y = boston.target
print(X.shape)

print(y.shape)
print(boston.feature_names)
print(boston.DESCR)
import pandas as pd

df = pd.DataFrame(X)

df.columns = boston.feature_names

df.head()

df.describe()


# Normalise this dataset

# Each feature must have 0 mean, unit variance

import numpy as np

u = np.mean(X,axis=0)

std = np.std(X,axis=0)

#print(u.shape,std.shape)
# Normalise the Data

X = (X-u)/std
# Normalised Data

pd.DataFrame(X[:5,:]).head()


# Plot Y vs any feature

import matplotlib.pyplot as plt



plt.style.use('seaborn')

plt.scatter(X[:,5],y)

plt.show()
X.shape, y.shape
ones = np.ones((X.shape[0],1))

X = np.hstack((ones,X))

print(X.shape)
print(X[3])


# X - Matrix ( m x n)

# x - Vector (Single Example with n features)



def hypothesis(x,theta):

    y_ = 0.0

    n = x.shape[0]

    for i in range(n):

        y_  += (theta[i]*x[i])

    return y_



def error(X,y,theta):

    e = 0.0

    m = X.shape[0]

    

    for i in range(m):

        y_ = hypothesis(X[i],theta)

        e += (y[i] - y_)**2

        

    return e/m



def gradient(X,y,theta):

    m,n = X.shape

    

    grad = np.zeros((n,))

    

    # for all values of j

    for j in range(n):

        #sum over all examples

        for i in range(m):

            y_ = hypothesis(X[i],theta)

            grad[j] += (y_ - y[i])*X[i][j]

    # Out of the loops

    return grad/m



def gradient_descent(X,y,learning_rate=0.1,max_epochs=300):

    m,n = X.shape

    theta = np.zeros((n,))

    error_list = []

    

    for i in range(max_epochs):

        e = error(X,y,theta)

        error_list.append(e)

        

        # Gradient Descent

        grad = gradient(X,y,theta)

        for j in range(n):

            theta[j] = theta[j] - learning_rate*grad[j]

        

    return theta,error_list


import time

start = time.time()

theta,error_list = gradient_descent(X,y)

end = time.time()

print("Time taken is ", end-start)


print(theta)
plt.plot(error_list)

plt.show()
y_ = []

m = X.shape[0]



for i in range(m):

    pred = hypothesis(X[i],theta)

    y_.append(pred)

y_ = np.array(y_)
def r2_score(y,y_):

    num = np.sum((y-y_)**2)

    denom = np.sum((y- y.mean())**2)

    score = (1- num/denom)

    return score*100
# SCORE

r2_score(y,y_)

def hypothesis(X,theta):

    return np.dot(X,theta)



def error(X,y,theta):

    e = 0.0

    y_ = hypothesis(X,theta)

    e = np.sum((y-y_)**2)

    

    return e/m

    

def gradient(X,y,theta):

    

    y_ = hypothesis(X,theta)

    grad = np.dot(X.T,(y_ - y)) 

    m = X.shape[0]

    return grad/m



def gradient_descent(X,y,learning_rate = 0.1,max_iters=300):

    

    n = X.shape[1]

    theta = np.zeros((n,))

    error_list = []

    

    for i in range(max_iters):

        e = error(X,y,theta)

        error_list.append(e)

        

        #Gradient descent

        grad = gradient(X,y,theta)

        theta = theta - learning_rate*grad

        

    return theta,error_list


start = time.time()

theta,error_list = gradient_descent(X,y)

end = time.time()

print("Time taken by Vectorized Code",end-start)

theta
plt.plot(error_list)

plt.show()
y_ = hypothesis(X,theta)

r2_score(y,y_)