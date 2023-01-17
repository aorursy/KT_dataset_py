import collections



import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.manifold import MDS

from sklearn.preprocessing import MinMaxScaler
trainData = pd.read_csv('../input/train.csv')



trainPriceRange = trainData["price_range"]

trainData = trainData.drop("price_range", axis=1)
counter = collections.Counter(trainPriceRange)

counterTable = [[x, y] for x, y in counter.items()]

counterDataFrame = pd.DataFrame(data = counterTable, columns = ['priceRange', 'count'])



sns.set(style="whitegrid")

ax = sns.barplot(x="priceRange", y="count", data=counterDataFrame)
embedding = MDS(n_components = 2)

trainDataTransformed = embedding.fit_transform(trainData[:500])
ax = sns.scatterplot(x = trainDataTransformed[:, 0], y = trainDataTransformed[:, 1],

                     hue = trainPriceRange[:500])
scaler = MinMaxScaler()

scaler.fit(trainData)

trainDataScaled = scaler.transform(trainData)
kmeans = KMeans(n_clusters=4, random_state=0).fit(trainDataScaled)
metrics.adjusted_rand_score(trainPriceRange.values, kmeans.labels_) 
ax = sns.scatterplot(x = trainDataTransformed[:, 0], y = trainDataTransformed[:, 1],

                     hue = kmeans.labels_[:500])