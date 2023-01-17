# imports

import numpy as np

import matplotlib.pyplot as plt
# generate random data-set

np.random.seed(0)

noise = np.random.rand(100, 1)

x = np.random.rand(100, 1)

y = 3 * x + 15 + noise

# y=ax+b Target function  a=3, b=15
# plot

plt.scatter(x,y,s=10)

plt.xlabel('x')

plt.ylabel('y')

plt.show()
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)



model.fit(x, y)



pred = model.predict(x)



plt.scatter(x, y,s=10)

plt.plot(x, pred, color="r")