# imports

import numpy as np

import matplotlib.pyplot as plt



# generate random data-set

np.random.seed(0)

x = np.random.rand(100, 1)

y = 2 + 3 * x + np.random.rand(100, 1)



# plot

plt.scatter(x,y,s=10)

plt.xlabel('x')

plt.ylabel('y')

plt.show()