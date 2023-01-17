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
#BXDframe=pd.read_excel("/kaggle/input/bxd-info/RadialMazeData.xlsx")

#BXDframe=pd.read_excel("/kaggle/input/bxd-info/openfield.xlsx")

BXDframe=pd.read_excel("/kaggle/input/bxd-info/BrainBodyData.xlsx")

BXDframe['sex']=BXDframe['sex'].replace("M", 0)

BXDframe['sex']=BXDframe['sex'].replace("F", 1)

for i in range(1,400):

    BXDframe['Strain']=BXDframe['Strain'].replace("BXD"+str(i), i)

    BXDframe=BXDframe[BXDframe["Strain"] != 'C57']

    BXDframe=BXDframe[BXDframe["Strain"] != 'DBA']

print(BXDframe.columns)
BXDframe=BXDframe[BXDframe["Brainw"] != '.']

BXDframe=BXDframe.drop(columns="nr")

BXDframe2=BXDframe

from sklearn.preprocessing import scale

BXDframe=scale(BXDframe)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(BXDframe)

X_pca = pca.transform(BXDframe)



ex_variance=np.var(X_pca,axis=0)

ex_variance_ratio = ex_variance/np.sum(ex_variance)

print (ex_variance_ratio)

import matplotlib.pyplot as plt

import seaborn as sns

print(X_pca)

cdict={0:'red',1:'green'}

labl={0:'Malignant',1:'Benign'}

marker={0:'*',1:'o'}

plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.xlabel('component 1')

plt.ylabel('component 2')



from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.4, min_samples=20).fit(X_pca)

print(db)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)
from sklearn import metrics

print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)



unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X_pca[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X_pca[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()
print(X_pca.shape)

print(BXDframe2)

label2 = BXDframe2["sex"]

print(label2)