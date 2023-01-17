import numpy as np

import matplotlib.pyplot as plt

mean = [0, 100]

cov = [[1, 0.5], 

       [0.5, 1]]

x, y = np.random.multivariate_normal(mean, cov, 1000).T

plt.plot(x, y, 'x')

plt.axis('equal')
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
x = x.reshape(-1,1)

regr.fit(x, y)
y_hat = regr.predict(x)
plt.scatter(x, y,  color='black')

plt.plot(x, y_hat, color='blue', linewidth=3)
e = y - y_hat
plt.scatter(x, e)
plt.scatter(y, e)