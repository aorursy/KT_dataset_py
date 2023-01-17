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
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (12,9)

sns.set()



from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:,0], X[:,1])
# try with KMeans: result won't be perfect because this algorithm classifies well only spherical clusters



from sklearn.cluster import KMeans



km = KMeans(n_clusters=2)

y_km = km.fit_predict(X)



plt.scatter(X[:,0], X[:,1], c=y_km, cmap="viridis")
# Agglomerative Clustering doens't work here as well:



from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters=2, linkage="complete")



y_hc = hc.fit_predict(X)



plt.scatter(X[:,0], X[:,1], c=y_hc, cmap="viridis")
# try with dbscan

from sklearn.cluster import DBSCAN



#min_samples has to be >= dataset's dimensions + 1

#here the datasets has 2 dimension, so let's try 3



dbscan = DBSCAN(eps=0.25, min_samples=3)

y_dbscan = dbscan.fit_predict(X)



plt.scatter(X[:,0], X[:,1], c=y_dbscan, cmap="viridis")



# see how dbscan fits well with cluster where shape is not spheric