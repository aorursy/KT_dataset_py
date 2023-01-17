import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
def score(delta_clipped,sigma_clipped):
    score = - np.sqrt(2) * delta_clipped / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return score
delta_values = [x for x in range(0,350,10)]
sigma_values = [x for x in range(70,8000,10)]
from pylab import meshgrid
X,Y = meshgrid(delta_values, sigma_values)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, score(X,Y), rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=10)

ax.view_init(20, 310)


plt.show()
