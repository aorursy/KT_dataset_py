import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/indian-food-101/indian_food.csv")
df.head()
df.info()
df.describe()
df.isnull().any()
df.dropna()
df2 = df.dropna()
df2["diet"] = df2["diet"].astype('category')

df2["flavor_profile"] = df2["flavor_profile"].astype('category')

df2["course"] = df2["course"].astype('category')

df2["state"] = df2["state"].astype('category')

df2["region"] = df2["region"].astype('category')
df2["diet"] = df2["diet"].cat.codes

df2["flavor_profile"] = df2["flavor_profile"].cat.codes

df2["course"] = df2["course"].cat.codes

df2["state"] = df2["state"].cat.codes

df2["region"] = df2["region"].cat.codes
df2.head()
df3 = df2.drop(columns = ['name','ingredients'])
X = df3.values

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents1 = pca.fit_transform(X)
principalComponents1
PCA_dataset1 = pd.DataFrame(data = principalComponents1, columns = ['component1', 'component2'] )

PCA_dataset1.head()
principal_component1 = PCA_dataset1['component1']

principal_component2 = PCA_dataset1['component2']
plt.figure()

plt.figure(figsize=(10,10))

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.title('2 Component PCA')

plt.scatter(PCA_dataset1['component1'], PCA_dataset1['component2'])
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 30, init = 'k-means++', random_state = 1)

y_kmeans = kmeans.fit_predict(principalComponents1)
from matplotlib import colors as mcolors
plt.scatter(principalComponents1[y_kmeans == 0, 0], principalComponents1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(principalComponents1[y_kmeans == 1, 0], principalComponents1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(principalComponents1[y_kmeans == 2, 0], principalComponents1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(principalComponents1[y_kmeans == 3, 0], principalComponents1[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(principalComponents1[y_kmeans == 4, 0], principalComponents1[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(principalComponents1[y_kmeans == 5, 0], principalComponents1[y_kmeans == 5, 1], s = 100, c = 'limegreen', label = 'Cluster 6')

plt.scatter(principalComponents1[y_kmeans == 6, 0], principalComponents1[y_kmeans == 6, 1], s = 100, c = 'lavender', label = 'Cluster 7')

plt.scatter(principalComponents1[y_kmeans == 7, 0], principalComponents1[y_kmeans == 7, 1], s = 100, c = 'black', label = 'Cluster 8')

plt.scatter(principalComponents1[y_kmeans == 8, 0], principalComponents1[y_kmeans == 8, 1], s = 100, c = 'dimgray', label = 'Cluster 9')

plt.scatter(principalComponents1[y_kmeans == 9, 0], principalComponents1[y_kmeans == 9, 1], s = 100, c = 'silver', label = 'Cluster 10')

plt.scatter(principalComponents1[y_kmeans == 10, 0], principalComponents1[y_kmeans == 10, 1], s = 100, c = 'gainsboro', label = 'Cluster 11')

plt.scatter(principalComponents1[y_kmeans == 11, 0], principalComponents1[y_kmeans == 11, 1], s = 100, c = 'white', label = 'Cluster 12')

plt.scatter(principalComponents1[y_kmeans == 12, 0], principalComponents1[y_kmeans == 12, 1], s = 100, c = 'whitesmoke', label = 'Cluster 13')

plt.scatter(principalComponents1[y_kmeans == 13, 0], principalComponents1[y_kmeans == 13, 1], s = 100, c = 'rosybrown', label = 'Cluster 14')

plt.scatter(principalComponents1[y_kmeans == 14, 0], principalComponents1[y_kmeans == 14, 1], s = 100, c = 'indianred', label = 'Cluster 15')

plt.scatter(principalComponents1[y_kmeans == 15, 0], principalComponents1[y_kmeans == 15, 1], s = 100, c = 'firebrick', label = 'Cluster 16')

plt.scatter(principalComponents1[y_kmeans == 16, 0], principalComponents1[y_kmeans == 16, 1], s = 100, c = 'red', label = 'Cluster 17')

plt.scatter(principalComponents1[y_kmeans == 17, 0], principalComponents1[y_kmeans == 17, 1], s = 100, c = 'mistyrose', label = 'Cluster 18')

plt.scatter(principalComponents1[y_kmeans == 18, 0], principalComponents1[y_kmeans == 18, 1], s = 100, c = 'salmon', label = 'Cluster 19')

plt.scatter(principalComponents1[y_kmeans == 19, 0], principalComponents1[y_kmeans == 19, 1], s = 100, c = 'darksalmon', label = 'Cluster 20')

plt.scatter(principalComponents1[y_kmeans == 20, 0], principalComponents1[y_kmeans == 20, 1], s = 100, c = 'coral', label = 'Cluster 21')

plt.scatter(principalComponents1[y_kmeans == 21, 0], principalComponents1[y_kmeans == 21, 1], s = 100, c = 'orangered', label = 'Cluster 22')

plt.scatter(principalComponents1[y_kmeans == 22, 0], principalComponents1[y_kmeans == 22, 1], s = 100, c = 'sienna', label = 'Cluster 23')

plt.scatter(principalComponents1[y_kmeans == 23, 0], principalComponents1[y_kmeans == 23, 1], s = 100, c = 'seashell', label = 'Cluster 24')

plt.scatter(principalComponents1[y_kmeans == 24, 0], principalComponents1[y_kmeans == 24, 1], s = 100, c = 'chocolate', label = 'Cluster 25')

plt.scatter(principalComponents1[y_kmeans == 25, 0], principalComponents1[y_kmeans == 25, 1], s = 100, c = 'saddlebrown', label = 'Cluster 26')

plt.scatter(principalComponents1[y_kmeans == 26, 0], principalComponents1[y_kmeans == 26, 1], s = 100, c = 'sandybrown', label = 'Cluster 27')

plt.scatter(principalComponents1[y_kmeans == 27, 0], principalComponents1[y_kmeans == 27, 1], s = 100, c = 'peachpuff', label = 'Cluster 28')

plt.scatter(principalComponents1[y_kmeans == 28, 0], principalComponents1[y_kmeans == 28, 1], s = 100, c = 'peru', label = 'Cluster 29')

plt.scatter(principalComponents1[y_kmeans == 29, 0], principalComponents1[y_kmeans == 29, 1], s = 100, c = 'bisque', label = 'Cluster 30')
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(principalComponents1, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Compounds')

plt.ylabel('Euclidean distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc2 = AgglomerativeClustering(n_clusters = 30, affinity = 'euclidean', linkage = 'ward')

y_hc2 = hc2.fit_predict(principalComponents1)
plt.scatter(principalComponents1[y_hc2 == 0, 0], principalComponents1[y_hc2 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(principalComponents1[y_hc2 == 1, 0], principalComponents1[y_hc2 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(principalComponents1[y_hc2 == 2, 0], principalComponents1[y_hc2 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(principalComponents1[y_hc2 == 3, 0], principalComponents1[y_hc2 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(principalComponents1[y_hc2 == 4, 0], principalComponents1[y_hc2 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(principalComponents1[y_hc2 == 5, 0], principalComponents1[y_hc2 == 5, 1], s = 100, c = 'limegreen', label = 'Cluster 6')

plt.scatter(principalComponents1[y_hc2 == 6, 0], principalComponents1[y_hc2 == 6, 1], s = 100, c = 'lavender', label = 'Cluster 7')

plt.scatter(principalComponents1[y_hc2 == 7, 0], principalComponents1[y_hc2 == 7, 1], s = 100, c = 'black', label = 'Cluster 8')

plt.scatter(principalComponents1[y_hc2 == 8, 0], principalComponents1[y_hc2 == 8, 1], s = 100, c = 'dimgray', label = 'Cluster 9')

plt.scatter(principalComponents1[y_hc2 == 9, 0], principalComponents1[y_hc2 == 9, 1], s = 100, c = 'silver', label = 'Cluster 10')

plt.scatter(principalComponents1[y_hc2 == 10, 0], principalComponents1[y_hc2 == 10, 1], s = 100, c = 'gainsboro', label = 'Cluster 11')

plt.scatter(principalComponents1[y_hc2 == 11, 0], principalComponents1[y_hc2 == 11, 1], s = 100, c = 'white', label = 'Cluster 12')

plt.scatter(principalComponents1[y_hc2 == 12, 0], principalComponents1[y_hc2 == 12, 1], s = 100, c = 'whitesmoke', label = 'Cluster 13')

plt.scatter(principalComponents1[y_hc2 == 13, 0], principalComponents1[y_hc2 == 13, 1], s = 100, c = 'rosybrown', label = 'Cluster 14')

plt.scatter(principalComponents1[y_hc2 == 14, 0], principalComponents1[y_hc2 == 14, 1], s = 100, c = 'indianred', label = 'Cluster 15')

plt.scatter(principalComponents1[y_hc2 == 15, 0], principalComponents1[y_hc2 == 15, 1], s = 100, c = 'firebrick', label = 'Cluster 16')

plt.scatter(principalComponents1[y_hc2 == 16, 0], principalComponents1[y_hc2 == 16, 1], s = 100, c = 'red', label = 'Cluster 17')

plt.scatter(principalComponents1[y_hc2 == 17, 0], principalComponents1[y_hc2 == 17, 1], s = 100, c = 'mistyrose', label = 'Cluster 18')

plt.scatter(principalComponents1[y_hc2 == 18, 0], principalComponents1[y_hc2 == 18, 1], s = 100, c = 'salmon', label = 'Cluster 19')

plt.scatter(principalComponents1[y_hc2 == 19, 0], principalComponents1[y_hc2 == 19, 1], s = 100, c = 'darksalmon', label = 'Cluster 20')

plt.scatter(principalComponents1[y_hc2 == 20, 0], principalComponents1[y_hc2 == 20, 1], s = 100, c = 'coral', label = 'Cluster 21')

plt.scatter(principalComponents1[y_hc2 == 21, 0], principalComponents1[y_hc2 == 21, 1], s = 100, c = 'orangered', label = 'Cluster 22')

plt.scatter(principalComponents1[y_hc2 == 22, 0], principalComponents1[y_hc2 == 22, 1], s = 100, c = 'sienna', label = 'Cluster 23')

plt.scatter(principalComponents1[y_hc2 == 23, 0], principalComponents1[y_hc2 == 23, 1], s = 100, c = 'seashell', label = 'Cluster 24')

plt.scatter(principalComponents1[y_hc2 == 24, 0], principalComponents1[y_hc2 == 24, 1], s = 100, c = 'chocolate', label = 'Cluster 25')

plt.scatter(principalComponents1[y_hc2 == 25, 0], principalComponents1[y_hc2 == 25, 1], s = 100, c = 'saddlebrown', label = 'Cluster 26')

plt.scatter(principalComponents1[y_hc2 == 26, 0], principalComponents1[y_hc2 == 26, 1], s = 100, c = 'sandybrown', label = 'Cluster 27')

plt.scatter(principalComponents1[y_hc2 == 27, 0], principalComponents1[y_hc2 == 27, 1], s = 100, c = 'peachpuff', label = 'Cluster 28')

plt.scatter(principalComponents1[y_hc2 == 28, 0], principalComponents1[y_hc2 == 28, 1], s = 100, c = 'peru', label = 'Cluster 29')

plt.scatter(principalComponents1[y_hc2 == 29, 0], principalComponents1[y_hc2 == 29, 1], s = 100, c = 'bisque', label = 'Cluster 30')