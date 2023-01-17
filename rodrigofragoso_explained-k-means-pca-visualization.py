import pandas as pd

pd.set_option('display.max_columns', None)

pd.options.display.float_format = "{:.2f}".format



import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



from tqdm import tqdm
data=pd.read_csv('/kaggle/input/wine-dataset-for-clustering/wine-clustering.csv')
data.head()
data.info()
data.describe()
data.skew()
sns.set(style='white',font_scale=1.3, rc={'figure.figsize':(20,20)})

ax=data.hist(bins=20,color='red' )
data.plot( kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False,color='black')

plt.show()
data.isnull().sum().sort_values(ascending=False).head()
from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()

data_cluster=data.copy()

data_cluster[data_cluster.columns]=std_scaler.fit_transform(data_cluster)
data_cluster.describe()
from sklearn.decomposition import PCA

pca_2 = PCA(2)

pca_2_result = pca_2.fit_transform(data_cluster)



print ('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))
sns.set(style='white', rc={'figure.figsize':(9,6)},font_scale=1.1)



plt.scatter(x=pca_2_result[:, 0], y=pca_2_result[:, 1], color='red',lw=0.1)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.title('Data represented by the 2 strongest principal components',fontweight='bold')

plt.show()
import sklearn.cluster as cluster



inertia = []

for i in tqdm(range(2,10)):

    kmeans = cluster.KMeans(n_clusters=i,

               init='k-means++',

               n_init=15,

               max_iter=500,

               random_state=17)

    kmeans.fit(data_cluster)

    inertia.append(kmeans.inertia_)
from sklearn.metrics import silhouette_score



silhouette = {}

for i in tqdm(range(2,10)):

    kmeans = cluster.KMeans(n_clusters=i,

               init='k-means++',

               n_init=15,

               max_iter=500,

               random_state=17)

    kmeans.fit(data_cluster)

    silhouette[i] = silhouette_score(data_cluster, kmeans.labels_, metric='euclidean')
sns.set(style='white',font_scale=1.1, rc={'figure.figsize':(12,5)})



plt.subplot(1, 2, 1)



plt.plot(range(2,len(inertia)+2), inertia, marker='o',lw=2,ms=8,color='red')

plt.xlabel('Number of clusters')

plt.title('K-means Inertia',fontweight='bold')

plt.grid(True)



plt.subplot(1, 2, 2)



plt.bar(range(len(silhouette)), list(silhouette.values()), align='center',color= 'red',width=0.5)

plt.xticks(range(len(silhouette)), list(silhouette.keys()))

plt.grid()

plt.title('Silhouette Score',fontweight='bold')

plt.xlabel('Number of Clusters')





plt.show()
kmeans = cluster.KMeans(n_clusters=3,random_state=17,init='k-means++')

kmeans_labels = kmeans.fit_predict(data_cluster)



centroids = kmeans.cluster_centers_

centroids_pca = pca_2.transform(centroids)



pd.Series(kmeans_labels).value_counts()
data2=data.copy()

data2['Cluster']=kmeans_labels



aux=data2.columns.tolist()

aux[0:len(aux)-1]



for cluster in aux[0:len(aux)-1]:

    grid= sns.FacetGrid(data2, col='Cluster')

    grid.map(plt.hist, cluster,color='red')
centroids_data=pd.DataFrame(data=std_scaler.inverse_transform(centroids), columns=data.columns)

centroids_data.head()
sns.set(style='white', rc={'figure.figsize':(9,6)},font_scale=1.1)



plt.scatter(x=pca_2_result[:, 0], y=pca_2_result[:, 1], c=kmeans_labels, cmap='autumn')

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],

            marker='x', s=169, linewidths=3,

            color='black', zorder=10,lw=3)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.title('Clustered Data (PCA visualization)',fontweight='bold')

plt.show()