import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(rc={'figure.figsize':(35,15)})
matplotlib.rc('figure', figsize=(15, 7))
matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 
matplotlib.rc('axes', titlesize=17)
from mpl_toolkits import mplot3d
import copy  
def encoding(data):
    #colonnes catégorielles
    df = copy.deepcopy(data)
    for i in df.select_dtypes(include=['object']).columns:
        list_unique = set(df[i].unique())
        dict_pro = dict(zip(list_unique,np.arange(len(list_unique))))
        df[i] = df[i].map(dict_pro)
    return df

def plot_acp(data,pca,i):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.set(font_scale=1)
    plt.step(range(data.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
    sns.barplot(np.arange(1,data.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
    plt.xlim(0, data.shape[1]/i)

    ax.set_xticklabels([s if int(s.get_text())%5 == 0 else '' for s in ax.get_xticklabels()], rotation=90)

    plt.ylabel('Explained variance', fontsize = 14)
    plt.xlabel('Principal components', fontsize = 14)
    plt.legend(loc='best', fontsize = 13); 
column = pd.read_csv('../input/columns.csv')
column.head()
column.shape
data = pd.read_csv('../input/responses.csv')
data.head()
data.shape
data.select_dtypes(include=['object']).columns
data = encoding(data)
data.head()
#data = data.fillna(0)
data = data.dropna()
data.shape
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data)
pca_samples = pca.transform(data)
plot_acp(data,pca,4)
plt.scatter(pca_samples[:, 0], pca_samples[:, 1])
plt.title('Distribution des deux premières features de l ACP')
plt.show()
ax = plt.axes(projection='3d')
ax.scatter3D(pca_samples[:, 0], pca_samples[:, 1], pca_samples[:, 2])
plt.title('Distribution des trois premières features de l ACP')
plt.show()
from sklearn.preprocessing import StandardScaler
data_scale = StandardScaler().fit_transform(data)
from sklearn.decomposition import PCA
pcaS = PCA()
pcaS.fit(data_scale)
pca_samples_scale = pcaS.transform(data_scale)
plot_acp(data_scale,pcaS,1)
plt.scatter(pca_samples_scale[:, 0], pca_samples_scale[:, 1])
plt.title('Distribution des deux premières features de l ACP')
plt.show()
ax = plt.axes(projection='3d')
ax.scatter3D(pca_samples_scale[:, 0], pca_samples_scale[:, 1], pca_samples_scale[:, 2])
plt.title('Distribution des trois premières features de l ACP')
plt.show()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

for n_clusters in range(2,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
kmeans = KMeans(init='k-means++', n_clusters = 3, n_init=30)
kmeans.fit(data)
clusters_ok = kmeans.predict(data)
pd.Series(clusters_ok).value_counts()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

for n_clusters in range(2,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(data_scale)
    clusters = kmeans.predict(data_scale)
    silhouette_avg = silhouette_score(data_scale, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
from sklearn.cluster import DBSCAN
list_score=[]
for i in range(1,30):
    kmeans = KMeans(init='k-means++', n_clusters = 3, n_init=30)
    kmeans.fit(pca_samples[:,0:i])
    clusters = kmeans.predict(pca_samples[:,0:i])
    list_score.append(silhouette_score(pca_samples[:,0:i], clusters))
fig, ax = plt.subplots()
plt.plot(np.arange(1,30),list_score)
ax.set_xticks(np.arange(1,30))
plt.title('Silhouette score en fonction du nombre de composantes choisies')
plt.show()
print('MAX :',round(max(list_score),2),'de silhouette score pour les',
      list_score.index(max(list_score))+1,'premières composantes.')
#Clusters choisies : clusters construit avec KNN et la première composante principale
kmeans = KMeans(init='k-means++', n_clusters = 3, n_init=30)
kmeans.fit(pca_samples[:,0:1])
clusters_ok = kmeans.predict(pca_samples[:,0:1])
pd.Series(clusters_ok).value_counts()
data['cluster']=clusters_ok
data.head()
data_cluster = data.groupby(['cluster']).mean()
data_cluster
data_cluster.iloc[:,0:19].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,19:31].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,31:47].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,47:63].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,63:76].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,76:90].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,90:105].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,105:119].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,119:133].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,133:140].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,140:143].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()
data_cluster.iloc[:,143:].T.plot(kind='bar')
plt.tick_params(labelsize=15)
plt.show()


