#imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# creating a regresssion set

from sklearn.datasets import make_regression

X,Y = make_regression(n_features=1,noise=10,n_samples=1000)
plt.xlabel('X')

plt.ylabel(' Y')

plt.scatter(X,Y,s=5)

plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
#fit the model

lr.fit(X,Y)
print('coefficient = ',lr.coef_)

print('intercept =',lr.intercept_)
pred = lr.predict(X)
plt.scatter(X,Y,label='training')

plt.scatter(X,pred,label='prediction')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.legend()

plt.show()
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()

sgd.fit(X,Y)

print('coefficient = ',sgd.coef_)

print('intercept =',sgd.intercept_)

pred = sgd.predict(X)
plt.scatter(X,Y,label='training')

plt.scatter(X,pred,label='prediction')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.legend()

plt.show()
from sklearn.linear_model import Ridge



ridge = Ridge(alpha=.1)

lr = LinearRegression()

ridge.fit([[0, 0], [0, 0], [1, 1]],  [0, .1, 1])

lr.fit([[0, 0], [0, 0], [1, 1]],  [0, .1, 1])
print('coefficient = ',ridge.coef_)

print('intercept =',ridge.intercept_)
#adding outlinear

outliers = Y[950:] - 600

Y_Out = np.append(Y[:950],outliers)

plt.scatter(X,Y_Out)

plt.show()
lr = LinearRegression()

lr.fit(X,Y_Out)
pred_out = lr.predict(X)


plt.scatter(X,Y_Out,label='actual')

plt.scatter(X,pred_out,label='prediction with outliers')

plt.scatter(X,pred,s=5,c='k', label='prediction without outlier')

plt.legend()

plt.title('Linear Regression')


lr.coef_
#data generation

X,y,w = make_regression(n_samples=10, n_features=10,coef=True,random_state=1,bias=3.5)
w

alphas = np.logspace(-6,6,200)

alphas[:20]
coefs=[]

for a in alphas:

    ridge = Ridge(alpha=a,fit_intercept=False)

    ridge.fit(X,y)

    coefs.append(ridge.coef_)



ax =plt.gca()

ax.plot(alphas,coefs)

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization')

plt.show()

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=.1)

lasso.fit([[0, 0], [0, 0], [1, 1]],  [0, .1, 1])
lasso.coef_
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=.1)

en.fit([[0, 0], [0, 0], [1, 1]],  [0, .1, 1])
en.coef_
from sklearn.datasets import make_blobs

X,y = make_blobs(n_features=2, n_samples=1000, cluster_std=2,centers=2)

plt.scatter(X[:,0],X[:,1],c=y,s=10)

plt.show()

h = .02

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X,y)
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=y,s=10)

plt.show()