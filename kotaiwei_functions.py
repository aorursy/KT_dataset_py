# This Python 3 environment to learn Machine Learning 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # create visualize graph
import math # Math Function

x = np.arange(-math.pi,math.pi,0.1)
plt.plot(x,np.vectorize(math.sin)(x))
plt.show()
