# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

X=np.array([[10, 10], [10,11],[12,10],[10,11],[-10,10],[-10,9],[-10,11],[-10,8],[-11,9]])

y=[0,0,0,0,1,1,1,1,1]

sns.scatterplot(x=X[:,0], y=X[:,1])

plt.savefig('data.png', dpi=400)
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans



cmap_light = ListedColormap(['orange', 'cornflowerblue'])

cmap_bold = ListedColormap(['darkorange', 'darkblue'])



h=0.2

clf =  KMeans(n_clusters=2, random_state=0).fit(X)





# Plot the decision boundary

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot the nodes

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=20,

            edgecolor='k')

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())



plt.savefig('result.png', dpi=400)

plt.show()