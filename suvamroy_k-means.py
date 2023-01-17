import pandas as pd

data = pd.read_csv("../input/iris/Iris.csv")

data
data.columns
del data['Id']
data

del data['Species']

data
data
data.drop(['Id'],axis=1,inplace= True)

data
from sklearn.cluster import KMeans

import numpy as np

Kmeans = KMeans(n_clusters=4, random_state=0)

Kmeans.fit(data)
Kmeans.labels_
X = data.data[:, :2]

y = data.target