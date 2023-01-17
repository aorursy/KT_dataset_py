# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base= pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
base.drop_duplicates() #remove duplicates
base.dropna()
base.info() #feature output
base2 = base.drop('date', axis = 1)
base2 = base2.drop('id', axis = 1)
base2 = base2.drop('zipcode', axis = 1)
base2
# Check if all elements are 0 in the waterfront and view columns (Answer: No)
(base2['waterfront'] == 0).all()
(base2['view'] == 0).all()
base2.describe()
fig=plt.figure(figsize=(14,14))
sns.heatmap(base2.corr(), square = True, annot = True, linewidths = .5)
plt.title("Correlation matrix:")
plt.show()
# First way of scaling (explicit way)
work_base = base2.values[:, 0:]
work_base = (work_base - work_base.mean(axis = 0)) / work_base.std(axis = 0)
work_base
# Standard deviation check:
work_base.std(axis = 0)
# Second way of scaling (implicit way)
from sklearn.preprocessing import StandardScaler # For Mx=0 and Dx=1
X = base2.values[:, 0:]
work_base = StandardScaler().fit_transform(X)
work_base
# Standard deviation check:
work_base.std(axis = 0)
# Now we can check hierarchical clustering :)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# We conduct cluster analysis, where 'ward' - the distance between clusters and 
# 'euclidean' - the distance between objects in one cluster
link = linkage(work_base, method = 'ward', metric = 'euclidean') # 1 obj, 2 obj, dist, number of objects in a cluster
print(link[:5])
fig = plt.figure(figsize=(18,18))
dend = dendrogram(link, orientation = 'right', labels = base.id.values)
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=12)
ax.tick_params(axis='y', which='major', labelsize=12)
#Add the "cluster" column, in which we write down the number of the cluster, into which the object fell, 
# the distance between the objects = 225
base['cluster'] = fcluster(link, 230, criterion = 'distance')
base.groupby('cluster').mean()
# Displaying where and which clusters are
for i, group in base.groupby('cluster'):
    print('=' * 80)
    print('cluster ', i)
    print(group)
# However, we have a lot of objects, so it is more convenient to use the KMeans method
from sklearn.cluster import KMeans
k_means = KMeans(init = 'k-means++', n_clusters = 2, random_state = 42, n_init = 12).fit(work_base)
# Clustering result
clust_K_means = k_means.labels_
print('Meтки кластеров: %s ' % clust_K_means)
base['cluster'] = clust_K_means
base.groupby('cluster').mean()
# Cluster center coordinates
k_means.cluster_centers_
# If we have a new object, what cluster should we refer to?
new_item = [[1.1, 4, 3, 3000, 40000, 1, 0, 1, 3, 9, 2000, 700, 1000, 190, 1000000, 58, 120, 3000]]
k_means.predict(new_item)
# So how many clusters should you take?
K = range (1, 9) # determined the number of clusters we are interested in
k_means = [KMeans(n_clusters = k,init = 'k-means++', random_state = 42).fit(work_base) for k in K]
# inertia_ - returns the sum of the distances from each data point to the center of the closest cluster
dist = [model.inertia_ for model in k_means] # Our 'quality criterion'
# Plot the elbow
plt.plot(K, dist, marker = 'o')
plt.xlabel('K')
plt.ylabel('Sum of distances')
plt.title('The Elbow Method showing the optimal K')
plt.show()
base.groupby('cluster').size()
# We can visualize this
plt.scatter(work_base[:,2], work_base[:,0], c = clust_K_means.astype(np.float), alpha = 0.5)
plt.title('K_means Clustering')
plt.xlabel('bathrooms', fontsize = 16)
plt.ylabel('price', fontsize = 16)
plt.show()
plt.scatter(work_base[:,2], work_base[:,0], c = fcluster(link, 230, criterion = 'distance'), alpha = 0.5)
plt.title('Hierarchical clustering (ward)')
plt.xlabel('bathrooms', fontsize = 16)
plt.ylabel('price', fontsize = 16)
plt.show()
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage= 'complete').fit_predict(work_base)
print('Meтки кластеров: %s ' % ac)
plt.scatter(work_base[:,2], work_base[:,0], c = fcluster(link, 230, criterion = 'distance'), alpha = 0.5)
plt.title('Hierarchical clustering (complete)')
plt.xlabel('bathrooms', fontsize = 16)
plt.ylabel('price', fontsize = 16)
plt.show()
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage= 'single').fit_predict(work_base)
print('Meтки кластеров: %s ' % ac)
plt.scatter(work_base[:,2], work_base[:,0], c = fcluster(link, 230, criterion = 'distance'), alpha = 0.5)
plt.title('Hierarchical clustering (single)')
plt.xlabel('bathrooms', fontsize = 16)
plt.ylabel('price', fontsize = 16)
plt.show()