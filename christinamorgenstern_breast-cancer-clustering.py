# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import cm
import seaborn as sns; sns.set()
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.cluster import adjusted_rand_score
# load dataset and explore the first rows
df = pd.read_csv('/kaggle/input/breast-cancer-gene-expression-cumida/Breast_GSE45827.csv')
df.head()
# retrieve number of rows and columns in the dataset
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns in the breast cancer data set')
# check for missing values in dataset
df.isnull().sum()
# check for unique ID identifiers
print(f"The total ids are {df['samples'].count()}, from those the unique ids are {df['samples'].value_counts().shape[0]} ")
# check for label distribution
label_count = df['type'].value_counts()
label_count
# visualize distribution of labels
fig = plt.figure(figsize=(7, 5))
df['type'].value_counts().plot(kind='bar')
plt.xticks(rotation=45)
plt.ylabel('Number of occurences', fontsize=12, fontweight='bold')
plt.xlabel('Sample type', fontsize=12, fontweight='bold')
plt.title('Distribution of label types in breast cancer data', fontsize=14, fontweight='bold')
# assign labels to variable y
y = df['type']
y
# select feature data for clustering
data = df.iloc[:,2:].values
data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data
### k-Means Clustering
# Calculate the cluster errors for clusters from 1 to 15
cluster_range = range( 1, 20 )
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans(num_clusters, n_init = 10 )
  clusters.fit(scaled_data)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:20]
# Elbow plot
plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.xlabel('Number of clusters', fontsize=12, fontweight='bold')
plt.ylabel('Cluster error', fontsize=12, fontweight='bold')
plt.title('Elbow plot for determining number of clusters', fontsize=14, fontweight='bold')
plt.savefig('elbowplot.png')
# instantiate KMeans object
km = KMeans(n_clusters=6, random_state=0)
# predict the cluster labels
labels = km.fit_predict(scaled_data)
km.cluster_centers_.shape
centroids = km.cluster_centers_
print(centroids)
# print cluster labels
print(labels)
## creating a new dataframe only for labels and converting it into categorical variable
df_labels = pd.DataFrame(km.labels_ , columns = list(['label']))

df_labels['label'] = df_labels['label'].astype('category')
# Joining the label dataframe with the original data frame. 
df_labeled = df.join(df_labels)
df_labeled.head()
df_labeled['label'].value_counts()
print('Distortion: %.2f' % km.inertia_)
def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation
permutation = find_permutation(6, y, km.labels_)
print(permutation)
new_labels = [ permutation[label] for label in km.labels_]   # permute the labels
print("Accuracy score is", accuracy_score(y, new_labels))
# plot confusion matrix
mat = confusion_matrix(y, new_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=permutation,
            yticklabels=permutation)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.savefig('confustion_matrix_1')
# create silhoutte plot
cluster_labels = np.unique(labels)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(scaled_data,
                                      labels,
                                      metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[labels == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(i) / n_clusters)
     plt.barh(range(y_ax_lower, y_ax_upper),
              c_silhouette_vals,
              height=1.0,
              edgecolor='none',
              color=color)
     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
             color="red",
             linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
#plt.show()
plt.savefig('silhoutte_plot_1.png')
kmeansSilhouette_Score = metrics.silhouette_score(data, labels, metric='euclidean')
print(kmeansSilhouette_Score)
rand_index = adjusted_rand_score(labels_true = y, labels_pred = labels)
print('The Rand index is', round(rand_index, 2))
# In order to find the number of dimensions explaining most of the variety in the data, plot cumulative explained variance
pca_plot = PCA().fit(scaled_data)
plt.plot(np.cumsum(pca_plot.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
tsne = TSNE(random_state=0)
tsne_result = tsne.fit_transform(data)
xi = tsne_result[:, 0]
yi = tsne_result[:, 1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=xi, y=yi,
    hue=y,
    legend="full",
    alpha=1
)
plt.savefig('t-SNE_plot.png')
tsne_scaled = TSNE(random_state=0)
tsne_result_scaled = tsne.fit_transform(scaled_data)
xi_scaled = tsne_result_scaled[:, 0]
yi_scaled = tsne_result_scaled[:, 1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=xi_scaled, y=yi_scaled,
    hue=y,
    legend="full",
    alpha=1
)
km_tsne = KMeans(n_clusters = 6, random_state=0)
# predict the cluster labels
labels_tsne = km_tsne.fit_predict(tsne_result)
labels_tsne.size
labels_tsne
## creating a new dataframe only for labels and converting it into categorical variable
df_labels_tsne = pd.DataFrame(km_tsne.labels_ , columns = list(['label']))
df_labels_tsne['label'] = df_labels_tsne['label'].astype('category')
df_labels_tsne.head()
df_labels_tsne['label'].value_counts()
# silhouette plot
cluster_labels = np.unique(labels_tsne)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(tsne_result,
                                      labels_tsne,
                                      metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[labels == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(i) / n_clusters)
     plt.barh(range(y_ax_lower, y_ax_upper),
              c_silhouette_vals,
              height=1.0,
              edgecolor='none',
              color=color)
     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
             color="red",
             linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
#plt.show()
plt.savefig('silhoutte_plot_2.png')
kmeansSilhouette_Score = metrics.silhouette_score(tsne_result, labels_tsne, metric='euclidean')
kmeansSilhouette_Score
permutation = find_permutation(6, y, km_tsne.labels_)
print(permutation)
new_labels = [ permutation[label] for label in km_tsne.labels_]   # permute the labels
print("Accuracy score is", accuracy_score(y, new_labels))
rand_index = adjusted_rand_score(labels_true = y, labels_pred = labels_tsne)
print('The Rand index is', round(rand_index, 2))
import umap
import numba.targets
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(data)
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=clusterable_embedding[:, 0], y=clusterable_embedding[:, 1],
    hue=y,
    legend="full",
    alpha=1
)
plt.savefig('UMAP_plot.png')
km_umap = KMeans(n_clusters = 6)
# predict the cluster labels
labels_umap = km_umap.fit_predict(clusterable_embedding)
# silhouette plot
cluster_labels = np.unique(labels_umap)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(clusterable_embedding,
                                      labels_umap,
                                      metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[labels == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(i) / n_clusters)
     plt.barh(range(y_ax_lower, y_ax_upper),
              c_silhouette_vals,
              height=1.0,
              edgecolor='none',
              color=color)
     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
             color="red",
             linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
#plt.show()
plt.savefig('silhoutte_plot_3.png')
kmeansSilhouette_Score = metrics.silhouette_score(clusterable_embedding, labels_umap, metric='euclidean')
kmeansSilhouette_Score
permutation = find_permutation(6, y, km_umap.labels_)
print(permutation)
new_labels = [ permutation[label] for label in km_umap.labels_]   # permute the labels
print("Accuracy score is", accuracy_score(y, new_labels))
rand_index = adjusted_rand_score(labels_true = y, labels_pred = labels_umap)
print('The Rand index is', round(rand_index, 2))
