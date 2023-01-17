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
import scipy.stats
import matplotlib.pyplot as plt
x = np.random.normal(2.,1, size=2000)
plt.hist(x,bins=30)
plt.show()
x = np.random.uniform(size=50)
y = np.random.uniform(size=50)
X = np.random.multivariate_normal([0.5,0], [[2,1.5],[1.5,1]],size=200)
X[:3]
plt.scatter(X[:,0],X[:,1])
ax = plt.gca()
ax.set_aspect('equal')
xx,yy = np.mgrid[-5:5:100j,-5:5:100j]
xx.shape
yy.shape
plt.scatter(xx.ravel(), yy.ravel())
M = np.column_stack((xx.ravel(), yy.ravel()))
M.shape
pdf = scipy.stats.multivariate_normal.pdf(M,mean=[1,1],cov=[[2,-0.5],[-0.5,1]])
pdf.shape
plt.contourf(xx, yy, pdf.reshape(xx.shape))
scipy.stats.multivariate_normal.pdf([1,2],mean=[1,1],cov=[[2,-0.5],[-0.5,1]])
np.cov([[1,2,3],[-1,-0.5,-3]])
x1 = np.random.normal(loc=5, scale=2,size=1600)
x2 = np.random.normal(loc=0, scale=1,size=2500)
250/(250 + 160)
xc = np.concatenate((x1,x2))
xc.shape
plt.hist(xc, bins=50)
plt.show()
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2)
gm.fit(xc.reshape(-1,1))
gm.means_
gm.covariances_
gm.weights_

