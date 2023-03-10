%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.backends import backend_agg

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.gridspec import GridSpec

import seaborn as sns

from IPython.display import Image



mu,sigma=100,15

x = mu + sigma * np.random.randn(10000)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.ylabel('Probability')

plt.title('Histogram of IQ')

plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

plt.axis([40, 160, 0, 0.03])

print(x.size)

print(n)

print(bins)

print(batches)

plt.show()




