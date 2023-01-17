# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import seaborn as sns 

from scipy.stats import pearsonr

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
top_2017 = pd.read_csv("../input/top-tracks-of-2017/featuresdf.csv")

top_2018 = pd.read_csv('../input/top-spotify-tracks-of-2018/top2018.csv')
top_2017.info()


top_2018.info()
top_2017_key_mode = top_2017[['key','mode']]

top_2018_key_mode = top_2018[['key','mode']]
import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(top_2017_key_mode)
g = sns.pairplot(top_2018_key_mode)
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Key vs Mode")

ax = fig.add_subplot(221)

g = plt.hist(top_2017_key_mode[top_2017_key_mode['mode'] == 0]['key'],bins=12,width=0.5)

ax = fig.add_subplot(222)

g = plt.hist(top_2017_key_mode[top_2017_key_mode['mode'] == 1]['key'],bins=12,width=0.5)

ax = fig.add_subplot(223)

g = plt.hist(top_2018_key_mode[top_2018_key_mode['mode'] == 1]['key'],bins=12,width=0.5)

ax = fig.add_subplot(224)

g = plt.hist(top_2018_key_mode[top_2018_key_mode['mode'] == 1]['key'],bins=12,width=0.5)
top_2017_numerical = top_2017[['danceability', 'energy', 'loudness',

        'speechiness', 'acousticness', 'instrumentalness', 'liveness',

       'valence', 'tempo', 'duration_ms', 'time_signature']]



top_2018_numerical = top_2018[['danceability', 'energy','loudness',

        'speechiness', 'acousticness', 'instrumentalness', 'liveness',

       'valence', 'tempo', 'duration_ms', 'time_signature']]
from sklearn import manifold
n_neighbors = 5

n_components = 2



isomap = manifold.Isomap(n_neighbors, n_components)

mds    = manifold.MDS(n_components, max_iter=100, n_init=1)

se     = manifold.SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors)

tsne   = manifold.TSNE(n_components=n_components, init='pca', random_state=0)



Y_2017_isomap = isomap.fit_transform(top_2017_numerical)

Y_2017_mds    = mds.fit_transform(top_2017_numerical)

Y_2017_se     = se.fit_transform(top_2017_numerical)

Y_2017_tsne   = tsne.fit_transform(top_2017_numerical)



Y_2018_isomap = isomap.fit_transform(top_2018_numerical)

Y_2018_mds    = mds.fit_transform(top_2018_numerical)

Y_2018_se     = se.fit_transform(top_2018_numerical)

Y_2018_tsne   = tsne.fit_transform(top_2018_numerical)



fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2017")

ax = fig.add_subplot(221)

plt.scatter(Y_2017_isomap[:, 0], Y_2017_isomap[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2017_mds[:, 0], Y_2017_mds[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2017_se[:, 0], Y_2017_se[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2017_tsne[:, 0], Y_2017_tsne[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2018")

ax = fig.add_subplot(221)

plt.scatter(Y_2018_isomap[:, 0], Y_2018_isomap[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2018_mds[:, 0], Y_2018_mds[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2018_se[:, 0], Y_2018_se[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2018_tsne[:, 0], Y_2018_tsne[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
middle_point_2017 = np.mean(Y_2017_se[:, 0])

labels_2017 = Y_2017_se[:,0]>middle_point_2017



middle_point_2018 = np.mean(Y_2018_se[:, 0])

labels_2018 = Y_2018_se[:,0]>middle_point_2018
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Outlier song")

ax = fig.add_subplot(121)

plt.scatter(Y_2017_se[:, 0], Y_2017_se[:, 1], c = labels_2017,cmap=plt.cm.Spectral)

ax = fig.add_subplot(122)

plt.scatter(Y_2018_se[:, 0], Y_2018_se[:, 1], c = labels_2018,cmap=plt.cm.Spectral)
top_2017[Y_2017_se[:,0]<middle_point_2017][['name','artists']]
top_2018[Y_2018_se[:,0]<middle_point_2017][['name','artists']]
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2017")

ax = fig.add_subplot(221)

plt.scatter(Y_2017_isomap[:, 0], Y_2017_isomap[:, 1], c = labels_2017,cmap=plt.cm.Spectral)

plt.title("Isomap 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2017_mds[:, 0], Y_2017_mds[:, 1], c = labels_2017,cmap=plt.cm.Spectral)

plt.title("MDS 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2017_se[:, 0], Y_2017_se[:, 1],c = labels_2017, cmap=plt.cm.Spectral)

plt.title("SE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2017_tsne[:, 0], Y_2017_tsne[:, 1], c = labels_2017,cmap=plt.cm.Spectral)

plt.title("TSNE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2018")

ax = fig.add_subplot(221)

plt.scatter(Y_2018_isomap[:, 0], Y_2018_isomap[:, 1], c = labels_2018,cmap=plt.cm.Spectral)

plt.title("Isomap 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2018_mds[:, 0], Y_2018_mds[:, 1],  c = labels_2018, cmap=plt.cm.Spectral)

plt.title("MDS 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2018_se[:, 0], Y_2018_se[:, 1],  c = labels_2018, cmap=plt.cm.Spectral)

plt.title("SE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2018_tsne[:, 0], Y_2018_tsne[:, 1],  c = labels_2018, cmap=plt.cm.Spectral)

plt.title("TSNE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
top_2017_numerical.iloc[labels_2017,:]
Y_2017_isomap_outliers = isomap.fit_transform(top_2017_numerical.iloc[labels_2017,:])

Y_2017_mds_outliers    = mds.fit_transform(top_2017_numerical.iloc[labels_2017,:])

Y_2017_se_outliers     = se.fit_transform(top_2017_numerical.iloc[labels_2017,:])

Y_2017_tsne_outliers   = tsne.fit_transform(top_2017_numerical.iloc[labels_2017,:])



Y_2018_isomap_outliers = isomap.fit_transform(top_2018_numerical.iloc[labels_2018,:])

Y_2018_mds_outliers    = mds.fit_transform(top_2018_numerical.iloc[labels_2018,:])

Y_2018_se_outliers     = se.fit_transform(top_2018_numerical.iloc[labels_2018,:])

Y_2018_tsne_outliers   = tsne.fit_transform(top_2018_numerical.iloc[labels_2018,:])
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2017 Outliers")

ax = fig.add_subplot(221)

plt.scatter(Y_2017_isomap_outliers[:, 0], Y_2017_isomap_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2017_mds_outliers[:, 0], Y_2017_mds_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2017_se_outliers[:, 0], Y_2017_se_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2017_tsne_outliers[:, 0], Y_2017_tsne_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
g = sns.pairplot(top_2017_numerical.iloc[labels_2017,:])
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2018")

ax = fig.add_subplot(221)

plt.scatter(Y_2018_isomap_outliers[:, 0], Y_2018_isomap_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2018_mds_outliers[:, 0], Y_2018_mds_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2018_se_outliers[:, 0], Y_2018_se_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2018_tsne_outliers[:, 0], Y_2018_tsne_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
g = sns.pairplot(top_2018_numerical.iloc[labels_2018,:])
Y_2017_isomap_no_outliers = isomap.fit_transform(top_2017_numerical.iloc[~labels_2017,:])

Y_2017_mds_no_outliers    = mds.fit_transform(top_2017_numerical.iloc[~labels_2017,:])

Y_2017_se_no_outliers     = se.fit_transform(top_2017_numerical.iloc[~labels_2017,:])

Y_2017_tsne_no_outliers   = tsne.fit_transform(top_2017_numerical.iloc[~labels_2017,:])



Y_2018_isomap_no_outliers = isomap.fit_transform(top_2018_numerical.iloc[~labels_2018,:])

Y_2018_mds_no_outliers    = mds.fit_transform(top_2018_numerical.iloc[~labels_2018,:])

Y_2018_se_no_outliers     = se.fit_transform(top_2018_numerical.iloc[~labels_2018,:])

Y_2018_tsne_no_outliers   = tsne.fit_transform(top_2018_numerical.iloc[~labels_2018,:])
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2017 No Outliers")

ax = fig.add_subplot(221)

plt.scatter(Y_2017_isomap_no_outliers[:, 0], Y_2017_isomap_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2017_mds_no_outliers[:, 0], Y_2017_mds_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2017_se_no_outliers[:, 0], Y_2017_se_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2017_tsne_no_outliers[:, 0], Y_2017_tsne_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2017")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning for 2018 No Outliers")

ax = fig.add_subplot(221)

plt.scatter(Y_2018_isomap_no_outliers[:, 0], Y_2018_isomap_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("Isomap 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(222)

plt.scatter(Y_2018_mds_no_outliers[:, 0], Y_2018_mds_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("MDS 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(223)

plt.scatter(Y_2018_se_no_outliers[:, 0], Y_2018_se_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("SE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')

ax = fig.add_subplot(224)

plt.scatter(Y_2018_tsne_no_outliers[:, 0], Y_2018_tsne_no_outliers[:, 1], cmap=plt.cm.Spectral)

plt.title("TSNE 2018")

ax.xaxis.set_major_formatter(NullFormatter())

ax.yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')