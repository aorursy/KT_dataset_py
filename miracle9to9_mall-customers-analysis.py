import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
#Impoting Datasets
dataset = pd.read_csv("../input/Mall_Customers.csv")
dataset.head()
dataset.Genre.value_counts()
print(dataset.dtypes)
# Summary of Object Variables
dataset.describe(include=[np.object])

plt.figure(figsize = (15,5))
names  = dataset['Genre'].value_counts()[:2].index
values  = dataset['Genre'].value_counts()[:2].values
colors = ['cyan','yellowgreen']
#explode = []

plt.pie(values, labels = names , colors=colors ,startangle=90,shadow=True,autopct='%1.2f%%')
plt.axis('equal')
plt.show()
import seaborn as sb
sb.pairplot(dataset, hue='Spending Score (1-100)')
sb.pairplot(dataset, hue='Genre')
dataset.isnull().sum()
x = dataset.iloc[:,[3,4]].values

# Using dendograms to find optimal numbers of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s = 50, c = '#F72E6E', label = 'cluster 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], s = 50, c = '#60F6C2', label = 'cluster 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], s = 50, c = '#00FF3E', label = 'cluster 3')
plt.scatter(x[y_hc == 3,0], x[y_hc == 3,1], s = 50, c = '#B11EF5', label = 'cluster 4')
plt.scatter(x[y_hc == 4,0], x[y_hc == 4,1], s = 50, c = '#271276', label = 'cluster 5')
plt.title('Clusters of Customers Hierarchical')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5 , init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 250, c = 'grey', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()