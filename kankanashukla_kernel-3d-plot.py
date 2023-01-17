from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt,numpy as np
plt.clf()
fig = plt.figure(1)
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=plt.cm.jet)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=plt.cm.jet)

### strating here:

# normalize Z to [0..1]
Z=Z-Z.min()
Z=Z/Z.max()

#use zoom to make your data smoother
from scipy.ndimage.interpolation import zoom

#make data 5 times smoother
X=zoom(X,5)
Y=zoom(Y,5)
Z=zoom(Z,5)

#draw a surface at -100, using the facecolors command to color it with the values of Z
cset = ax.plot_surface(X, Y, np.zeros_like(Z)-100,facecolors=plt.cm.jet(Z),shade=False)


ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)    
plt.show()# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#https://plot.ly/python/3d-charts/