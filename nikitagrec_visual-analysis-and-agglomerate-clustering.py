import numpy as np

import pandas as pd

import seaborn as sns

import sklearn

from sklearn.decomposition import PCA as PCA

from sklearn.metrics import silhouette_samples

import matplotlib.pyplot as plt

from matplotlib import cm

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/Mall_Customers.csv')
print('shape = ', data.shape)

print(data.columns)
data_model = data.drop(['CustomerID'], axis=1)

data_model['Gender'] = data_model['Gender'].factorize()[0]

data.head()
sns.pairplot(data[['Gender', 'Age', 'Annual Income (k$)',

       'Spending Score (1-100)']]);
g = sns.PairGrid(data[['Gender', 'Age', 'Annual Income (k$)',

       'Spending Score (1-100)']])

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=8);
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3,3,1)

sns.barplot(y='Age',x='Gender', data=data);

ax2 = fig.add_subplot(3,3,2)

sns.barplot(y='Annual Income (k$)',x='Gender', data=data);

ax3 = fig.add_subplot(3,3,3)

sns.barplot(y='Spending Score (1-100)',x='Gender', data=data);
pca = PCA(n_components=2)

pca.fit(data_model)

Xpca = pca.transform(data_model)

sns.set()

plt.figure(figsize=(8,8))

plt.scatter(Xpca[:,0],Xpca[:,1], c='Red')

plt.show()
from sklearn.manifold import TSNE

tsn = TSNE()

res_tsne = tsn.fit_transform(data_model)

plt.figure(figsize=(8,8))

plt.scatter(res_tsne[:,0],res_tsne[:,1]);
from sklearn.cluster import AgglomerativeClustering as AggClus
clus_mod = AggClus(n_clusters=6)

assign = clus_mod.fit_predict(data_model)

plt.figure(figsize=(8,8))

sns.set(style='darkgrid',palette='muted')

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='Set1');
from scipy.cluster.hierarchy import dendrogram, ward

sns.set(style='white')

plt.figure(figsize=(10,7))

link = ward(res_tsne)

dendrogram(link)

ax = plt.gca()

bounds = ax.get_xbound()

ax.plot(bounds, [30,30],'--', c='k')

ax.plot(bounds,'--', c='k')

plt.show()
clus_mod = AggClus(n_clusters=6)

assign = clus_mod.fit_predict(data_model)

plt.figure(figsize=(8,8))

sns.set(style='darkgrid',palette='muted')

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='copper');
data_model['predict'] = pd.DataFrame(assign)
data_model.head(3)
fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(3,3,1)

sns.countplot(data['Gender'],hue=data_model['predict']);

ax2 = fig.add_subplot(3,3,2)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=data['Gender'], palette='copper');

ax3 = fig.add_subplot(3,3,3)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=data['Age'], palette='copper');
fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(3,3,1)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=data['Spending Score (1-100)'], palette='copper');

ax2 = fig.add_subplot(3,3,2)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=data['Annual Income (k$)'], palette='copper');
model = pd.DataFrame()

model['age'] = data_model['Age'].groupby(data_model['predict']).median()

model['annual income'] = data_model['Annual Income (k$)'].groupby(data_model['predict']).median()

model['spending score'] = data_model['Spending Score (1-100)'].groupby(data_model['predict']).median()

model.reset_index(inplace=True)
cluster_labels=np.unique(assign)

n_clusters = len(np.unique(assign))

silhouette_vals = silhouette_samples(res_tsne, assign, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0

yticks = []

plt.figure(figsize=(10,8))

for i , c in enumerate(cluster_labels):

        c_silhouette_vals = silhouette_vals[assign==c]

        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(float(i) / n_clusters)

        plt.barh(range(y_ax_lower,y_ax_upper),

                c_silhouette_vals,height=1.0,edgecolor='none',color=color)

        yticks.append((y_ax_lower+y_ax_upper) / 2)

        y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)



plt.axvline(silhouette_avg,color="red",linestyle= "--")

plt.yticks(yticks , cluster_labels + 1)

plt.ylabel ('Cluster')

plt.xlabel('Silhouette coefficient')
def clust_sill(num):

    fig = plt.figure(figsize=(25,20))

    ax1 = fig.add_subplot(3,3,1)



    clus_mod = AggClus(n_clusters=num)

    assign = clus_mod.fit_predict(data_model)

    sns.set(style='darkgrid',palette='muted')

    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

    sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='copper');

    cluster_labels=np.unique(assign)

    n_clusters = len(np.unique(assign))

    silhouette_vals = silhouette_samples(res_tsne, assign, metric='euclidean')



    y_ax_lower, y_ax_upper = 0, 0

    yticks = []

    ax2 = fig.add_subplot(3,3,2)

    for i , c in enumerate(cluster_labels):

        c_silhouette_vals = silhouette_vals[assign==c]

        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(float(i) / n_clusters)

        plt.barh(range(y_ax_lower,y_ax_upper),

                c_silhouette_vals,height=1.0,edgecolor='none',color=color)

        yticks.append((y_ax_lower+y_ax_upper) / 2)

        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)

    

    plt.title(str(num)+ ' Clusters')

    plt.axvline(silhouette_avg,color="red",linestyle= "--")

    plt.yticks(yticks , cluster_labels + 1)

    plt.ylabel ('Cluster')

    plt.xlabel('Silhouette coefficient')

clust_sill(3)

clust_sill(4)

clust_sill(5)

clust_sill(7)
model
fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(3,3,1)

sns.boxplot(y='Age',x='predict',data=data_model);

ax2 = fig.add_subplot(3,3,2)

sns.boxplot(y='Annual Income (k$)',x='predict',data=data_model);

ax3 = fig.add_subplot(3,3,3)

sns.boxplot(y='Spending Score (1-100)',x='predict',data=data_model);