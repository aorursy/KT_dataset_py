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
g=pd.read_csv("/kaggle/input/datascience/datascience/movies.csv")
g
x=pd.read_csv("/kaggle/input/datascience/datascience/ratings.csv")
x
ze=x.merge(g,on='movieId',how='left')
ze
vb=ze.groupby(['userId','title'])['rating'].mean()
vb
vb[1]['Toy Story (1995)']
nm=ze['title'].unique()
bv=ze['userId'].unique()
bv
za=pd.DataFrame(index=bv,columns=nm)
za
for i in nb:
    for j in nm:
        try:
            za.loc[i][j]=vb[i][j]
        except:
            za.loc[i][j]=0
nb=bv[600:700]
kj=za
za
from sklearn.preprocessing import StandardScaler
kj = StandardScaler().fit_transform(kj)
kj
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=320, min_samples=100).fit(kj)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
db
from sklearn import metrics
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = kj[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = kj[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
