# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score





data = pd.read_csv("../input/spinem.csv")

f1 = data['pelvic_incidence'].values

f2 = data['pelvic_radius'].values

f3 = data['thoracic_slope'].values

X = np.array(list(zip(f1, f2, f3)))

kmeans = KMeans(n_clusters = 5, random_state = 50)

model = kmeans.fit(X)

cluster_labels = kmeans.predict(X)

C = kmeans.cluster_centers_

sil = silhouette_score(X, cluster_labels, metric='euclidean',sample_size = len(data))

print(C)

print(sil)



fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1])

plt.scatter(C[:, 0], C[:, 1], marker='*', s=1000)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2])

ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker= '*', c='#050505',s=1000)
