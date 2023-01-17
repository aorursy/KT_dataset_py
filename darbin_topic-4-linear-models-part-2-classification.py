%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np
def sigma(z):

    return 1. / (1 + np.exp(-z))
xx = np.linspace(-10, 10, 1000)

plt.plot(xx, [sigma(x) for x in xx]);

plt.xlabel('z');

plt.ylabel('sigmoid(z)')

plt.title('Sigmoid function');