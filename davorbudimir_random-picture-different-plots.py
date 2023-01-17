# random picture, random mumber, different plots

import matplotlib.pyplot as plt
import numpy as np

# create data
m = 200
x = np.random.rand(m)
y = np.random.rand(m)
z = np.random.rand(m)
colors = np.random.rand(m)
 
# use the scatter function
plt.scatter(x, y, s=z*1000,c=colors)
plt.show()

# histogram
x = numpy.random.uniform(0.0, 100.0, 1000)

plt.hist(x, 100)
plt.show()

# histogram normal distribution
x = numpy.random.normal(5.0, 100.0, 100000)

plt.hist(x, 100)
plt.show()

# scatter plot
x = numpy.random.uniform(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()
