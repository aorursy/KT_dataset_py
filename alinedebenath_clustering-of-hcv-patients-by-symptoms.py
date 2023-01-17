import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import seaborn as sns

sns.set_style("whitegrid")
data = pd.read_csv("../input/hepatitis-c-virus-for-egyptian-patients-data-set/HCV-Egy-Data.csv")
data.shape
data.head()
data.columns
data.dtypes
data.isna().sum()
scaler = StandardScaler()

data_scale = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans

model = KMeans()

visualizer = KElbowVisualizer(model, k=(4,14))

visualizer.fit(data)

visualizer.poof()
kmeans = KMeans(n_clusters=8)

clusters = kmeans.fit(data_scale)

data['Cluster'] = clusters.predict(data_scale)

data_scale['Cluster'] = clusters.predict(data_scale)
data_scale['Cluster'].value_counts(sort=False)
from sklearn.decomposition import PCA

import numpy as np

pca = PCA()

pca.fit(data_scale)

pca_x = pca.transform(data_scale)

pca_df = pd.DataFrame([pca_x[:, 0], pca_x[:, 1]]).T

pca_df.columns = ['PC1', 'PC2']
pca_df = pd.concat([pca_df, data_scale['Cluster']], axis=1)

sns.lmplot('PC1', 'PC2', data=pca_df, hue='Cluster', fit_reg=False)
data.groupby('Cluster').median()
data_scale.drop(columns=['Cluster'], inplace=True)

data.drop(columns=['Cluster'], inplace=True)
from sklearn.metrics import silhouette_score

silhouette_score(data_scale, kmeans.predict(data_scale))
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.01)

dbscan.fit(data_scale)

data_scale['Cluster2'] = dbscan.labels_
pca_df = pd.concat([pca_df, data_scale['Cluster2']], axis=1)

sns.lmplot('PC1', 'PC2', data=pca_df, hue='Cluster2', fit_reg=False)
data_scale.drop(columns=['Cluster2'], inplace=True)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=8).fit(data_scale)

data['Cluster3']= gmm.fit_predict(data_scale)
pca_df = pd.concat([pca_df, data['Cluster3']], axis=1)

sns.lmplot('PC1', 'PC2', data=pca_df, hue='Cluster3', fit_reg=False)
data.groupby('Cluster3').median()
data.drop(columns=['Cluster3'], inplace=True)
silhouette_score(data_scale, gmm.predict(data_scale), metric='euclidean')