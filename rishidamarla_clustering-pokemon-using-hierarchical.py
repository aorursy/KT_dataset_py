# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
df.info()
df.describe()
# Dropping non-integer features.

df2 = df.drop(['Type 1', 'Type 2'], axis = 1) 
df2.isnull().sum()
# Dropping non-integer features.

df3 = df2.drop(['#', 'Name', 'Legendary'], axis = 1) 
df3.dtypes
X = df3.values

# Using the standard scaler method to standardize all of the features by converting them into values between -3 and +3.

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
# Implementing T-Distributed Stochastic Network Embedding.

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results1 = tsne.fit_transform(X)
tsne_results1
# Creating a dataframe featuring the two t-sne components that we acquired through t-SNE.

tsne_dataset1 = pd.DataFrame(data = tsne_results1, columns = ['component1', 'component2'] )

tsne_dataset1.head()
# Extracting the two features from above in order to add them to the dataframe.

tsne_component1 = tsne_dataset1['component1']

tsne_component2 = tsne_dataset1['component2']
# Visualizing the effects of the T-distributed Stochastic Neighbour Embedding.

plt.figure()

plt.figure(figsize=(10,10))

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.title('2 Component TSNE')

plt.scatter(tsne_component1, tsne_component2)
# Implementing a dendogram to visualize the euclidean distanced between each data point.

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(tsne_results1, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean distances')

plt.show()
df3.shape
# Implementing the Hierachical Clustering.

from sklearn.cluster import AgglomerativeClustering

hc2 = AgglomerativeClustering(n_clusters = 40, affinity = 'euclidean', linkage = 'ward')

y_hc2 = hc2.fit_predict(tsne_results1)
# Plotting the clusters.

plt.scatter(tsne_results1[y_hc2 == 0, 0], tsne_results1[y_hc2 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(tsne_results1[y_hc2 == 1, 0], tsne_results1[y_hc2 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(tsne_results1[y_hc2 == 2, 0], tsne_results1[y_hc2 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(tsne_results1[y_hc2 == 3, 0], tsne_results1[y_hc2 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(tsne_results1[y_hc2 == 4, 0], tsne_results1[y_hc2 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(tsne_results1[y_hc2 == 5, 0], tsne_results1[y_hc2 == 5, 1], s = 100, c = 'limegreen', label = 'Cluster 6')

plt.scatter(tsne_results1[y_hc2 == 6, 0], tsne_results1[y_hc2 == 6, 1], s = 100, c = 'lavender', label = 'Cluster 7')

plt.scatter(tsne_results1[y_hc2 == 7, 0], tsne_results1[y_hc2 == 7, 1], s = 100, c = 'black', label = 'Cluster 8')

plt.scatter(tsne_results1[y_hc2 == 8, 0], tsne_results1[y_hc2 == 8, 1], s = 100, c = 'dimgray', label = 'Cluster 9')

plt.scatter(tsne_results1[y_hc2 == 9, 0], tsne_results1[y_hc2 == 9, 1], s = 100, c = 'silver', label = 'Cluster 10')

plt.scatter(tsne_results1[y_hc2 == 10, 0], tsne_results1[y_hc2 == 10, 1], s = 100, c = 'gainsboro', label = 'Cluster 11')

plt.scatter(tsne_results1[y_hc2 == 11, 0], tsne_results1[y_hc2 == 11, 1], s = 100, c = 'white', label = 'Cluster 12')

plt.scatter(tsne_results1[y_hc2 == 12, 0], tsne_results1[y_hc2 == 12, 1], s = 100, c = 'whitesmoke', label = 'Cluster 13')

plt.scatter(tsne_results1[y_hc2 == 13, 0], tsne_results1[y_hc2 == 13, 1], s = 100, c = 'rosybrown', label = 'Cluster 14')

plt.scatter(tsne_results1[y_hc2 == 14, 0], tsne_results1[y_hc2 == 14, 1], s = 100, c = 'indianred', label = 'Cluster 15')

plt.scatter(tsne_results1[y_hc2 == 15, 0], tsne_results1[y_hc2 == 15, 1], s = 100, c = 'firebrick', label = 'Cluster 16')

plt.scatter(tsne_results1[y_hc2 == 16, 0], tsne_results1[y_hc2 == 16, 1], s = 100, c = 'red', label = 'Cluster 17')

plt.scatter(tsne_results1[y_hc2 == 17, 0], tsne_results1[y_hc2 == 17, 1], s = 100, c = 'mistyrose', label = 'Cluster 18')

plt.scatter(tsne_results1[y_hc2 == 18, 0], tsne_results1[y_hc2 == 18, 1], s = 100, c = 'salmon', label = 'Cluster 19')

plt.scatter(tsne_results1[y_hc2 == 19, 0], tsne_results1[y_hc2 == 19, 1], s = 100, c = 'darksalmon', label = 'Cluster 20')

plt.scatter(tsne_results1[y_hc2 == 20, 0], tsne_results1[y_hc2 == 20, 1], s = 100, c = 'coral', label = 'Cluster 21')

plt.scatter(tsne_results1[y_hc2 == 21, 0], tsne_results1[y_hc2 == 21, 1], s = 100, c = 'orangered', label = 'Cluster 22')

plt.scatter(tsne_results1[y_hc2 == 22, 0], tsne_results1[y_hc2 == 22, 1], s = 100, c = 'sienna', label = 'Cluster 23')

plt.scatter(tsne_results1[y_hc2 == 23, 0], tsne_results1[y_hc2 == 23, 1], s = 100, c = 'seashell', label = 'Cluster 24')

plt.scatter(tsne_results1[y_hc2 == 24, 0], tsne_results1[y_hc2 == 24, 1], s = 100, c = 'chocolate', label = 'Cluster 25')

plt.scatter(tsne_results1[y_hc2 == 25, 0], tsne_results1[y_hc2 == 25, 1], s = 100, c = 'saddlebrown', label = 'Cluster 26')

plt.scatter(tsne_results1[y_hc2 == 26, 0], tsne_results1[y_hc2 == 26, 1], s = 100, c = 'sandybrown', label = 'Cluster 27')

plt.scatter(tsne_results1[y_hc2 == 27, 0], tsne_results1[y_hc2 == 27, 1], s = 100, c = 'peachpuff', label = 'Cluster 28')

plt.scatter(tsne_results1[y_hc2 == 28, 0], tsne_results1[y_hc2 == 28, 1], s = 100, c = 'peru', label = 'Cluster 29')

plt.scatter(tsne_results1[y_hc2 == 29, 0], tsne_results1[y_hc2 == 29, 1], s = 100, c = 'bisque', label = 'Cluster 30')

plt.scatter(tsne_results1[y_hc2 == 30, 0], tsne_results1[y_hc2 == 30, 1], s = 100, c = 'linen', label = 'Cluster 31')

plt.scatter(tsne_results1[y_hc2 == 31, 0], tsne_results1[y_hc2 == 31, 1], s = 100, c = 'darkorange', label = 'Cluster 32')

plt.scatter(tsne_results1[y_hc2 == 32, 0], tsne_results1[y_hc2 == 32, 1], s = 100, c = 'burlywood', label = 'Cluster 33')

plt.scatter(tsne_results1[y_hc2 == 33, 0], tsne_results1[y_hc2 == 33, 1], s = 100, c = 'antiquewhite', label = 'Cluster 34')

plt.scatter(tsne_results1[y_hc2 == 34, 0], tsne_results1[y_hc2 == 34, 1], s = 100, c = 'tan', label = 'Cluster 35')

plt.scatter(tsne_results1[y_hc2 == 35, 0], tsne_results1[y_hc2 == 35, 1], s = 100, c = 'navajowhite', label = 'Cluster 36')

plt.scatter(tsne_results1[y_hc2 == 36, 0], tsne_results1[y_hc2 == 36, 1], s = 100, c = 'orange', label = 'Cluster 37')

plt.scatter(tsne_results1[y_hc2 == 37, 0], tsne_results1[y_hc2 == 37, 1], s = 100, c = 'oldlace', label = 'Cluster 38')

plt.scatter(tsne_results1[y_hc2 == 38, 0], tsne_results1[y_hc2 == 38, 1], s = 100, c = 'darkgoldenrod', label = 'Cluster 39')

plt.scatter(tsne_results1[y_hc2 == 39, 0], tsne_results1[y_hc2 == 39, 1], s = 100, c = 'goldenrod', label = 'Cluster 40')

plt.title('Clusters of Pokemon Characters')

plt.xlabel('t-SNE Component 1')

plt.ylabel('t-SNE Component 2')

plt.show()