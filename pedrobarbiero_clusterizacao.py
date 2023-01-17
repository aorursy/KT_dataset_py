import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

from sklearn.cluster import KMeans
df = pd.read_excel("../input/campeonato-brasileiro-full.xlsx")

df.head()
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df[['p1','p2']].values)

labels = kmeans.predict(df[['p1','p2']].values)

C = kmeans.cluster_centers_

print(labels,C)
df.plot.hist(y='p1')

df.plot.hist(y='p2')