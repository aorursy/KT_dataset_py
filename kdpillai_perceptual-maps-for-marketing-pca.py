import pandas as pd

df= pd.read_csv("../input/rintro-chapter8 (1).csv")

df.head()
from sklearn import preprocessing

df_s = df.drop(columns = ['brand'])

df_s = pd.DataFrame(preprocessing.normalize(df_s), columns=df_s.columns)

df_s = df_s.join(df['brand'])

df_s.head()
import seaborn as sns

corr = df_s.corr()

#sns.light_palette("purple")

sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 7), center = 0, vmin = -1, vmax = 1)
import numpy as np

import scipy

import scipy.cluster.hierarchy as sch



X = df_s.corr().values

d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances

L = sch.linkage(d, method='complete')

ind = sch.fcluster(L, 0.5*d.max(), 'distance')

columns = [df_s.columns.tolist()[i] for i in list((np.argsort(ind)))]

df_s = df_s.reindex(columns, axis=1)

df_s.head()
corr = df_s.corr()

#sns.light_palette("purple")

sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 7), center = 0, vmin = -1, vmax = 1)
df_m = df.groupby(['brand']).mean()

df_m.head()
ax = sns.heatmap(df_m,  cmap="YlOrRd",linewidths=0.4)
ax = sns.clustermap(df_m,  cmap="YlOrRd",linewidths=0.4)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

df_s = df.drop(columns = ['brand'])

df_s = pd.DataFrame(preprocessing.normalize(df_s), columns=df_s.columns)

df_s = df_s.join(df['brand'])

X = df_s.drop(columns = ['brand'])

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC1', 'PC2', 'PC3'])

principalDf.head()
df_pca = principalDf.join(df_s['brand'])

df_pca.head()
pca = PCA(n_components=9)

principalComponents = pca.fit_transform(X)

pca.explained_variance_ratio_
import matplotlib.pyplot as plt

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
from yellowbrick.datasets import load_credit

from yellowbrick.features.pca import PCADecomposition

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 





df_s = df.drop(columns = ['brand'])

df_s = pd.DataFrame(preprocessing.normalize(df_s), columns=df_s.columns)

df_s = df_s.join(df['brand'])

df_s['brand']= label_encoder.fit_transform(df_s['brand']) #Label encoding for the YellowBricks functions to work



X = df_s.drop(columns = ['brand'])

y = df_s['brand']









visualizer = PCADecomposition(scale=True, proj_features = True, color = None)

visualizer.fit_transform(X, y)

visualizer.show()
df_agg = df.groupby(['brand'],as_index = False).mean()







df_s = df_agg.drop(columns = ['brand'])

df_s = pd.DataFrame(preprocessing.normalize(df_s), columns=df_s.columns)

df_s = df_s.join(df_agg['brand'])

df_s['brand']= label_encoder.fit_transform(df_s['brand']) #Label encoding for the YellowBricks functions to work



X = df_s.drop(columns = ['brand'])

y = df_s['brand']





visualizer = PCADecomposition(scale=True, proj_features = True, color = None)

visualizer.fit_transform(X, y)

visualizer.show()


