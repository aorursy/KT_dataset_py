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
from sklearn.datasets import make_moons,make_circles

import numpy as np

from matplotlib import pyplot as plt
p,q=make_moons(n_samples=200,noise=0.1)
plt.scatter(p[:,0],p[:,1],c=q,)

plt.xlabel("P1")

plt.ylabel("P2")

plt.legend()

plt.show()
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(p)
centers=km.cluster_centers_

label=km.labels_
plt.scatter(p[:,0],p[:,1],c=label)

plt.scatter(centers[:,0],centers[:,1],color="red",marker="*")

plt.show()

# as we can see its preety bad 
from sklearn.cluster import DBSCAN
dbs=DBSCAN(eps=0.188,min_samples=2)
dbs.fit(p)
ypred=dbs.fit_predict(p)
plt.scatter(p[:,0],p[:,1],c=ypred)

plt.show()