import numpy as np

import seaborn as sns

import pandas as pd  

import matplotlib.pyplot as plt 

from sklearn.datasets import load_wine

from sklearn.cluster import KMeans



wine = load_wine()

#dict_keys(['data', 'target', 'feature_names', 'DESCR'])

dataset = pd.DataFrame(wine.data, columns=wine.feature_names)

#dataset['Type'] = wine.target

print(dataset.head())
#APPLY THE Z SCORE SCALE TO TEST SET

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

dataset = sc.fit_transform(dataset)
#-------------KMEANS-----------

kmeans3 = KMeans(n_clusters=3)

y_kmeans3 = kmeans3.fit_predict(dataset)

print(y_kmeans3)

kmeans3.cluster_centers_

statistics =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(dataset)

    #kmeans.fit(dataset)

    statistics.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))

plt.plot(range(1, 11), statistics)

plt.title('Scree Plot')

plt.xlabel('Number of Clusters')

plt.ylabel('Squared distances from nearest cluster centroid')

#plt.savefig("\Wine_scree.png", dpi=150)

plt.show()
Df_WITH_TARGET = pd.concat([pd.DataFrame(dataset,columns=wine.feature_names), pd.DataFrame(y_kmeans3,columns=['Type'])], axis = 1)



a = list(np.arange(9,13))

a.append(13)

selection = a.copy()

#plt.savefig("\Wine_3clusters.png", dpi=150)

plot_pair=sns.pairplot(data=Df_WITH_TARGET.iloc[:,selection],palette='deep',hue="Type")

#plt.savefig("\Wine_3clusters.png", dpi=250)
#---------Hierarchical clustering----------------

from scipy.cluster.hierarchy import dendrogram, linkage



Z = linkage(dataset, 'complete')



# set cut-off to 150

max_d = 6                # max_d as in max_distance 300



plt.figure(figsize=(25, 10))

plt.title('Wine Dataset Hierarchical Clustering Dendrogram Using complete Linkage')

plt.xlabel('Type')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=170,                  # Try changing values of p

    leaf_rotation=90.,      # rotates the x axis labels

    leaf_font_size=8.,      # font size for the x axis labels

)

plt.axhline(y=max_d, c='k')

#plt.savefig("\Wine_dendro.png", dpi=150)

plt.show()

from sklearn.cluster import AgglomerativeClustering



groups = AgglomerativeClustering(n_clusters=3,

                                 affinity='euclidean', linkage='complete')

pred=groups .fit_predict(dataset)

pred
from sklearn.metrics import confusion_matrix

wine.target

cm=confusion_matrix(wine.target, pred)

accuracy = np.trace(cm)/np.sum(cm)

print(cm)

print("accuracy:",accuracy)
