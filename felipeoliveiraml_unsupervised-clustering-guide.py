import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 



%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')



from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from scipy.cluster.hierarchy import fcluster, linkage, dendrogram

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN
path = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'

data = pd.read_csv(path)

data.head()
data.describe()
data.dtypes
plt.figure(figsize = (15 , 6))

count = 0 

for feature in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    count += 1

    plt.subplot(1,3,count)

    plt.subplots_adjust(hspace=0.5 , wspace=0.5)

    sns.distplot(data[feature], bins=20)

    plt.title('Distribuição {}'.format(feature))

plt.show()
plt.figure(figsize=(15,7))

count = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        count += 1

        plt.subplot(3,3, count)

        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        sns.regplot(x=x, y=y, data=data)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
# features 

X = data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',

       'Spending Score (1-100)']]



# One hot encoding 

encoding = OneHotEncoder(sparse=False)

X['Gender'] = encoding.fit_transform(X[['Gender']])





# StandardScaler 

scaler = StandardScaler()

X = scaler.fit_transform(X)





# K-Means 

model = KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state=42)

model.fit(X)





# labels 

labels = model.labels_





# centroids 

centroids = model.cluster_centers_





# números de clusters

print('Numbers of cluster: ', model.n_clusters)





# metrics 

print('Silhoutte:', silhouette_score(X, labels))

print('Davies-Bouldin:', davies_bouldin_score(X, labels))
# set of features 

X1 = data[['Age' , 'Spending Score (1-100)']]

inercia = []

for k in range(1 , 11):

    kmeans = KMeans(n_clusters=k,

                     init='k-means++',

                     n_init=10,

                     max_iter=100, 

                     random_state=42,

                     algorithm='elkan')

    kmeans.fit(X1)

    inercia.append(kmeans.inertia_)
# Curve Inertia X Number of clusters



plt.figure(figsize = (15 ,6))

plt.plot(np.arange(1 , 11), inercia, 'o')

plt.plot(np.arange(1 , 11) , inercia, '-', color='green',alpha = 0.5, lw=2)

plt.title('Inercia Vs Clusters')

plt.xlabel('Numbers of cluster')

plt.ylabel('Inercia')

plt.show()
# Defined cluster centroids

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)

kmeans.fit(X)

labels = kmeans.labels_

print('Silhoutte:', silhouette_score(X, labels))

print('Davies-Bouldin:', davies_bouldin_score(X, labels))
# Silhouete points 



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Silhouette coefficient', fontsize=15)

visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax)

visualizer.fit(X)

plt.tight_layout()
# Elbow method



fig, ax = plt.subplots(figsize=(10,5))

plt.title('KElbow', fontsize=15)

visualizer = KElbowVisualizer(KMeans(), k=(1,11))

visualizer.fit(X)

plt.tight_layout()
# with scikit-learn  

hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean')

hc.fit(X)

labels_hc = hc.labels_





# metrics

print('Silhoutte:', silhouette_score(X, labels_hc))

print('Davies-Bouldin:', davies_bouldin_score(X, labels_hc))
# with scipy 

distance_matrix = linkage(X, metric='euclidean')



# dendogram 

fig, ax = plt.subplots(figsize=(14,7))

dendograma = dendrogram(distance_matrix, ax=ax)

plt.show()
hierarquico = fcluster(distance_matrix, 4, criterion='maxclust')

print('Silhouette: {} '.format(silhouette_score(X,hierarquico)))
# DBSCAN 

dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean', algorithm='auto')

dbscan.fit(X)

labels_db = dbscan.labels_





print('Silhoutte:', silhouette_score(X, labels_db))

print('Davies-Bouldin:', davies_bouldin_score(X, labels_db))