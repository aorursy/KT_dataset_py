import numpy as np

import matplotlib.pyplot as plt



sample_size = 25

x = np.random.rand(sample_size)

y = np.random.rand(sample_size)

colors = np.random.rand(sample_size)

area = (30 * np.random.rand(sample_size))**2  

plt.scatter(x, y, s=area, c=colors, alpha=0.7)

plt.show()



# Source: 

# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

# https://matplotlib.org/3.1.1/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py