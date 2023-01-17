!pip install bqplot

from bqplot import pyplot as plt

import numpy as np



plt.figure(1, title='Line Chart')

np.random.seed(0)

n = 200

x = np.linspace(0.0, 10.0, n)

y = np.cumsum(np.random.randn(n))

plt.plot(x, y)

plt.show()