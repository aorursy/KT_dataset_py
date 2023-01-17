# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#loading the data
data = pd.read_csv(r'../input/ccdata/CC GENERAL.csv')
#importing required libraries
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action = 'ignore')
#taking a quick look at the dara
print('the shape of the data is:', data.shape)
data.head()
data.describe()
data.head()
data = data.drop(columns = 'CUST_ID', axis = 1)
#checking for the null values
data.isna().sum()
#finding a method to fill these values;
data = data.interpolate()
data.isna().sum()
#finding the correlation between them
data_correlate = data.corr()
plt.figure(figsize = (12, 9))
sns.heatmap(data_correlate, linecolor = 'black', linewidth = 1, annot = True)
plt.title('Correlation of credit card data\'s features')
plt.show()
#dealing with the outliers:
from scipy import stats
z = np.abs(stats.zscore(data))
print(z)
data_outlier_free = pd.DataFrame(data[(z < 3).all(axis=1)], columns = data.columns)
print(data_outlier_free.shape)
data_outlier_free.head()
#Now standardising the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_outlier_free)
data_standardized = pd.DataFrame(data_standardized, columns = data.columns)
data_standardized.head()
#Now for k means algorithm of clustering we need a method to know the number of clusters.
#using elbow method to determine the clusters
#consider upto 30 clusters
num_clusters = 30
n_inertias = np.zeros((1, num_clusters))
from sklearn.cluster import KMeans
for i in range(0, num_clusters):
    k_means = KMeans(i+1)
    k_means.fit(data_standardized)
    n_inertias[0, i] = k_means.inertia_
plt.figure(figsize =  (12, 5))
plt.plot(n_inertias.flatten())
plt.xticks(range(0, num_clusters, 1))
plt.title('Using elbow method to find number of clusters')
plt.xlabel('No. of clusters')
plt.ylabel('inertias')
plt.draw()
plt.figure(figsize =  (12, 5))
arrow_properties = dict(
    facecolor="black", width=0.5,
    headwidth=4, shrink=0.1)

plt.annotate(
    "abrupt change(cluster count is 7)", xy=(7, 55700),
    xytext=(11, 80000),
    arrowprops=arrow_properties)
plt.plot(n_inertias.flatten())
plt.xticks(range(0, num_clusters, 1))
plt.title('Using elbow method to find number of clusters')
plt.xlabel('No. of clusters')
plt.ylabel('inertias')
plt.show()
#now using average sillhoute's method:
from sklearn.metrics import silhouette_score
silhouette_scores = []
num_clusters_list = []
for n_clusters in range(2,30):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(data_standardized)
    centers = clusterer.cluster_centers_

    silhouette_scores.append(silhouette_score (data_standardized, preds, metric='euclidean'))
    num_clusters_list.append(n_clusters)
plt.figure(figsize = (12, 5))
plt.plot(num_clusters_list, silhouette_scores)
plt.xticks(range(0, 30, 1))
plt.title('Using average silhouette\'s method to find number of clusters')
plt.xlabel('No. of clusters')
plt.ylabel('average silhouette score')
plt.show()
plt.figure(figsize = (12, 5))
arrow_properties = dict(
    facecolor="black", width=0.5,
    headwidth=4, shrink=0.1)

plt.annotate(
    "highesh silhouette's average value(no. of clusters = 8)", xy=(8, 0.203),
    xytext=(11, 0.205),
    arrowprops=arrow_properties)
plt.plot(num_clusters_list, silhouette_scores)
plt.title('Using average silhouette\'s method to find number of clusters')
plt.xlabel('No. of clusters')
plt.ylabel('average silhouette score')
plt.xticks(range(0, 30, 1))
plt.show()

k_means_7 = KMeans(7)
k_means_7.fit(data_standardized)
k_means_7_labels = k_means_7.labels_
data_out_1 = pd.concat([data_outlier_free, pd.DataFrame({'clusters_kmeans7': k_means_7_labels})], axis = 1)
for cols in data_outlier_free:
    g = sns.FacetGrid(data_out_1, col = 'clusters_kmeans7')
    g.map(plt.hist, cols)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
affinities = 1 - cosine_similarity(data_standardized)

pca = PCA(2)
pca.fit(affinities)
see_PCA = pca.transform(affinities)
x, y = see_PCA[:, 0], see_PCA[:, 1]

colors = {0: 'red', 
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple',
          6: 'pink'}

names = {0: 'people with less credit card use', 
         1: 'customers with most dues', 
         2: 'those who prefer EMI\'s', 
         3: 'customers whop prefer a lot of advance', 
         4: 'High purchase rate',
         5: 'who use less credit and maintain less',
         6: 'who pay full amount at once'}
  
df = pd.DataFrame({'x': x, 'y':y, 'labels':k_means_7_labels}) 
groups = df.groupby('labels')

fig, ax = plt.subplots(figsize=(15, 10)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("PCA implementation to visualise kmeans output")
plt.show()
k_means_8 = KMeans(8)
k_means_8.fit(data_standardized)
k_means_8_labels = k_means_8.labels_
data_out_2 = pd.concat([data_out_1, pd.DataFrame({'clusters_kmeans8': k_means_8_labels})], axis = 1)
for cols in data_outlier_free:
    g = sns.FacetGrid(data_out_2, col = 'clusters_kmeans8')
    g.map(plt.hist, cols)
from sklearn.decomposition import FastICA
fast_ica = FastICA(2)
fast_ica.fit(affinities)
see_ICA = fast_ica.transform(affinities)
x, y = see_ICA[:, 0], see_ICA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5: 'purple',
          6: 'pink',
          7: 'brown'}

names = {0: 'people with less credit card use', 
         1: 'customers with most dues', 
         2: 'those who prefer EMI\'s', 
         3: 'customers whop prefer a lot of advance', 
         4: 'High purchase rate',
         5: 'who use less credit and maintain less',
         6: 'who pay full amount at once',
         7: 'people who doesnt use'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':k_means_8_labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(15, 10)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=7,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("ICA implementation of the kmeans (clusters = 8)")
plt.show()
#affinity propagation.
from sklearn.cluster import AffinityPropagation
affinity_propagation = AffinityPropagation(0.9)
affinity_propagation.fit(data_standardized)
affinity_propagation_labels = affinity_propagation.labels_
data_out_3 = pd.concat([data_out_2, pd.DataFrame({'affinity_propagation': affinity_propagation_labels})], axis = 1)
#agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agglomerative_clustering = AgglomerativeClustering(7)
agglomerative_clustering.fit(data_standardized)
agglomerative_clustering_labels = agglomerative_clustering.labels_
data_out_4 = pd.concat([data_out_3, pd.DataFrame({'agglomerative_clustering': agglomerative_clustering_labels})], axis = 1)
for cols in data_outlier_free:
    g = sns.FacetGrid(data_out_4, col ='agglomerative_clustering')
    g.map(plt.hist, cols)
from sklearn.decomposition import TruncatedSVD
truncated_svd = TruncatedSVD(2)
truncated_svd.fit(affinities)
see_SVD = truncated_svd.transform(affinities)
x, y = see_SVD[:, 0], see_SVD[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple',
          6: 'pink'}

names = {0: 'who make all type of purchases', 
         1: 'more people with due payments', 
         2: 'who purchases mostly in installments', 
         3: 'who take more cash in advance', 
         4: 'who make expensive purchases',
         5:'who don\'t spend much money',
         6: 'dont know'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label': agglomerative_clustering_labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()
#DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
dbscan.fit(data_standardized)
dbscan_labels = dbscan.labels_
data_out_5 = pd.concat([data_out_4, pd.DataFrame({'dbscan': dbscan_labels})], axis = 1)
data_out_5.dbscan.unique()
from sklearn.manifold import Isomap
isomap = Isomap(len(data_out_5.dbscan.unique()), 2)
isomap.fit(affinities)
X_isomap = isomap.transform(affinities)

from matplotlib.colors import to_rgba
from matplotlib import colors as mcolors
x, y = X_isomap[:, 0], X_isomap[:, 1]


colors = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'orange', 5:'pink', 6:'brown', 7:'black',
          8:'khaki', 9:'lime', 10:'purple',
          11:'cyan', 12:'navy', 13:'chocolate', 14:'wheat', 15:'teal', 16:'magenta', 17:'coral',
          18: 'royalblue',19: 'maroon',20: 'grey',
          21:'darkgreen', 22:'olivedrab', 23:'goldenrod', -1:'firebrick', 24:'sienna'}
df = pd.DataFrame({'x': x, 'y':y, 'label':dbscan_labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=25,
            color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("visualising dbscan using isomap")
plt.show()
#OPTICS
from sklearn.cluster import OPTICS
optics = OPTICS()
optics.fit(data_standardized)
optics_labels = optics.labels_
data_out_6 = pd.concat([data_out_5, pd.DataFrame({'optics': optics_labels})], axis = 1)
data_out_6.optics.unique()
#Spectral clustering:
from sklearn.cluster import SpectralClustering
spectral_clustering = SpectralClustering(7)
spectral_clustering.fit(data_standardized)
spectral_clustering_labels = spectral_clustering.labels_
data_out_7 = pd.concat([data_out_6, pd.DataFrame({'spectral_clustering': spectral_clustering_labels})], axis = 1)
from sklearn.manifold import LocallyLinearEmbedding
locally_linear_embedding = LocallyLinearEmbedding(7, 2)
locally_linear_embedding.fit(affinities)
see_LLE = locally_linear_embedding.transform(affinities)
x, y = see_LLE[:, 0], see_LLE[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple',
          6: 'pink'}

names = {0: 'who make all type of purchases', 
         1: 'more people with due payments', 
         2: 'who purchases mostly in installments', 
         3: 'who take more cash in advance', 
         4: 'who make expensive purchases',
         5:'who don\'t spend much money',
         6: 'dont know'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label': spectral_clustering_labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(15, 10)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("spectral clustering visualization using LLE")
plt.show()
#mean shift
from sklearn.cluster import MeanShift
mean_shift = MeanShift()
mean_shift.fit(data_standardized)
mean_shift_labels = mean_shift.labels_
data_out_8 = pd.concat([data_out_7, pd.DataFrame({'mean_shift': mean_shift_labels})], axis = 1)
data_out_8.mean_shift.unique()
for cols in data_outlier_free:
    g = sns.FacetGrid(data_out_8, col ='mean_shift')
    g.map(plt.hist, cols)
from sklearn.manifold import SpectralEmbedding
spectral_embedding = SpectralEmbedding(2)
X_spectral_embedding = spectral_embedding.fit_transform(affinities)
x, y = X_spectral_embedding[:, 0], X_spectral_embedding[:, 1]

colors = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'orange', 5:'pink', 6:'brown', 7:'black',
          8:'khaki', 9:'lime', 10:'purple'}

df = pd.DataFrame({'x': x, 'y':y, 'label':mean_shift_labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(15, 10)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10,
            color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("mean_shift visualization using spectral clustering")
plt.show()
