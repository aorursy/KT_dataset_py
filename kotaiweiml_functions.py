# This Python 3 simple function output

import numpy as np # linear algebra
import matplotlib.pyplot as plt # visualize graph
import math #math function

x = np.arange(-math.pi,math.pi,0.1)
plt.plot(x,np.vectorize(math.sin)(x))
plt.show()
