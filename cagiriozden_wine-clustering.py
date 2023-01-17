import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/winequality.csv")
df['color'].replace({'white': 0, 'red': 1},inplace = True)

df.head()
df.describe()
sns.set()

df.hist(figsize=(10,10), color='green')

plt.show()
plt.figure(figsize=(12,12))

plt.title('Feature correlation', size=15)

sns.heatmap(df.astype(float).corr(),vmax=1.0, square=True, annot=True)
col = np.where(df.quality>7,'green',np.where(df.quality<5,'red','blue'))

plt.scatter(df['alcohol'] ,df['quality'],color=col,alpha = .5)

plt.show()
from sklearn.cluster import KMeans

inertia=np.empty(7)

for i in range(1,7):

    kmeans = KMeans(n_clusters=i, random_state=0).fit(df['quality'].values.reshape(-1,1))

    inertia[i] = kmeans.inertia_



plt.plot(range(0,7),inertia,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()
kmeans1 = KMeans(n_clusters=3, random_state=0)

clusters = kmeans1.fit_predict(df['quality'].values.reshape(-1,1))

df["cluster"] = clusters

df.head()



plt.scatter(df.quality[df.cluster == 0 ],df.pH[df.cluster == 0 ],color = "red")

plt.scatter(df.quality[df.cluster == 1 ],df.pH[df.cluster == 1 ],color = "green")

plt.scatter(df.quality[df.cluster == 2 ],df.pH[df.cluster == 2 ],color = "blue")

plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(df,method = 'ward')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)

plt.show()