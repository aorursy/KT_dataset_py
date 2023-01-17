from scipy.stats import beta

import matplotlib.pyplot as plt

import numpy as np

a =  2

b = 4

x = np.arange (0.01, 1, 0.01)

y = beta.pdf(x,a,b)

plt.plot(x,y)
