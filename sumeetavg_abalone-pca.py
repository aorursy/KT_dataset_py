import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns

sns.set()
df1 = pd.read_csv("../input/abalone.csv", header=1)

df1.head(5)
df1.drop(['M'], axis=1, inplace=True)

 
X = df1.iloc[:, 1:]
X.head()
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc = StandardScaler()
scaled_data = preprocessing.scale(X)

minmax_scaler = preprocessing.MinMaxScaler(feature_range =(0,1))

data_minmax  = minmax_scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=None)

X_sc = sc.fit_transform(X)

pca.fit(data_minmax)

np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100.)

plt.xlabel('number of components')

plt.ylabel('cummulative explained variance');
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5) #n_clusters=3,5

kmeans.fit(X)
y_kmeans = kmeans.predict(X)

from sklearn.metrics import silhouette_score

sse_ = []

for k in range(2, 8):

    kmeans = KMeans(n_clusters=k).fit(X)

    sse_.append([k, silhouette_score(X, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
