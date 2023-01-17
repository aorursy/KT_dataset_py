# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Libraries for visualization

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn import preprocessing

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from scipy.spatial import distance
mouse = pd.read_excel('../input/Data_Cortex_Nuclear.xls')

mouse.head()
m = mouse.drop(['MouseID','Genotype','Treatment','Behavior','class'],axis=1)

m.head()

#pravime da bide nenadgleduvano
m = m.fillna(m.mean())

columns = list(m.columns)

data = m.values

#gi poplnuvame praznite so mean
m.isnull().sum()
fig = plt.figure(figsize=(15, 50))

fig.subplots(77//2+1, ncols=2)

for feat_i in range(77): 

    ax = plt.subplot(77//2+1,2, feat_i+1)

    plt.title(columns[feat_i]) 

    sns.distplot(data[:,feat_i], color = "navy")

plt.show()

#sekoj atribut kako e raspredelen
m_scaled = m.subtract(m.mean())

m_scaled.mean(axis=0)

#centralizirame data okolku meanot za da bidat site vo slicen rang
columns1 = list(m_scaled.columns)

data1 = m_scaled.values

fig = plt.figure(figsize=(15, 70))

fig.subplots(77//2+1, ncols=2)

for feat_i in range(77): 

    ax = plt.subplot(77//2+1,2, feat_i+1)

    plt.title(columns1[feat_i]) 

    sns.distplot(data1[:,feat_i], color = "navy")

plt.show()
m.mean(axis=0)

#sporedba so gore
m
#pca na necentralizirana data, zimajki gi prvite 3 sopstveni vektori

pca = PCA(n_components=3)

m_pca = pca.fit_transform(m.values)



m['pca-one'] = m_pca[:,0]

m['pca-two'] = m_pca[:,1]

m['pca-three'] = m_pca[:,2]



pca.explained_variance_ratio_
#2d vizuelizacija, koja ne dava bash mngou info

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    palette=sns.color_palette("hls", 10),

    data=m,

    legend="full",

    alpha=0.3

)
#so centralizran df, se gleda deka ke ima potreba od pogolemo k no seushte nejasno

pca = PCA(n_components=3)

ms_pca = pca.fit_transform(m_scaled.values)



m_scaled['pca-one'] = ms_pca[:,0]

m_scaled['pca-two'] = ms_pca[:,1]

m_scaled['pca-three'] = ms_pca[:,2]



plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    palette=sns.color_palette("hls", 10),

    data=m_scaled,

    legend="full",

    alpha=0.3

)
#3d pca so necentralizirana data

ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=m["pca-one"], 

    ys=m["pca-two"], 

    zs=m["pca-three"], 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.show()
#t-sne

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(m)
#2d vizuelizacija so necentralizirana data

m['tsne-2d-one'] = tsne_results[:,0]

m['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    palette=sns.color_palette("hls", 10),

    data=m,

    legend="full",

    alpha=0.3

)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsnes_results = tsne.fit_transform(m_scaled)

m_scaled['tsne-2d-one'] = tsnes_results[:,0]

m_scaled['tsne-2d-two'] = tsnes_results[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    palette=sns.color_palette("hls", 10),

    data=m_scaled,

    legend="full",

    alpha=0.3

)
#m_scaled

fig1 = plt.figure(figsize=(9, 9))

wcv = {}

bcv = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k,max_iter=1000).fit(m_scaled)

    wcv[k] = kmeans.inertia_ 

    centers = kmeans.cluster_centers_

    BCV = 0

    for i in range(len(centers)):

        for j in range(len(centers)):

            BCV += distance.euclidean(centers[i], centers[j])**2

    if(k==1):

        bcv[1] = 0

    else:

        bcv[k] = BCV/(k*(k-1))*100

plt.plot(list(wcv.keys()), list(wcv.values()), label="Within Cluster Distance (WCV)")

plt.plot(list(bcv.keys()), list(bcv.values()), label="Between Cluster Distance (BCV)")

plt.xlabel("Number of clusters K")

plt.legend()

plt.show()
#klasteri vrz baza na necentrlizirana data

fig = plt.figure(figsize=(9, 9))

wcv = {}

bcv = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k,max_iter=1000).fit(m)

    wcv[k] = kmeans.inertia_ 

    centers = kmeans.cluster_centers_

    BCV = 0

    for i in range(len(centers)):

        for j in range(len(centers)):

            BCV += distance.euclidean(centers[i], centers[j])**2

    if(k==1):

        bcv[1] = 0

    else:

        bcv[k] = BCV/(k*(k-1))*100

plt.plot(list(wcv.keys()), list(wcv.values()), label="Within Cluster Distance (WCV)")

plt.plot(list(bcv.keys()), list(bcv.values()), label="Between Cluster Distance (BCV)")

plt.xlabel("Number of clusters K")

plt.legend()

plt.show()
#klasteri vrz ms_pca

fig1 = plt.figure(figsize=(9, 9))

wcv = {}

bcv = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k,max_iter=1000).fit(ms_pca)

    wcv[k] = kmeans.inertia_ 

    centers = kmeans.cluster_centers_

    BCV = 0

    for i in range(len(centers)):

        for j in range(len(centers)):

            BCV += distance.euclidean(centers[i], centers[j])**2

    if(k==1):

        bcv[1] = 0

    else:

        bcv[k] = BCV/(k*(k-1))*100

plt.plot(list(wcv.keys()), list(wcv.values()), label="Within Cluster Distance (WCV)")

plt.plot(list(bcv.keys()), list(bcv.values()), label="Between Cluster Distance (BCV)")

plt.xlabel("Number of clusters K")

plt.legend()

plt.show()
ms_pca
pca_mouse=pd.DataFrame(data=ms_pca[0:,0:],

                       index=[i for i in range(ms_pca.shape[0])],

                       columns=['f'+str(i) for i in range(ms_pca.shape[1])])

pca_mouse
fig = plt.figure(figsize=(50, 10))

#Fitting the PCA algorithm with our Data

pca = PCA().fit(m_scaled),

#Plotting the Cumulative Summation of the Explained Variance

plt.plot(np.cumsum(pca.explained_variance_ratio_))

ax.set_xticks(np.arange(1, 77+1, 1.0))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 9,)

y_kmeans = kmeans.fit_predict(pca_mouse)

#beginning of  the cluster numbering with 1 instead of 0

y_kmeans1=y_kmeans

y_kmeans1=y_kmeans+1

# New Dataframe called cluster

cluster = pd.DataFrame(y_kmeans1)

# Adding cluster to the Dataset1

pca_mouse['cluster'] = cluster

#Mean of clusters

kmeans_mean_cluster = pd.DataFrame(round(pca_mouse.groupby('cluster').mean(),1))

fig = plt.figure(figsize=(10, 10))

plt.scatter(pca_mouse.iloc[:, 0], pca_mouse.iloc[:, 1], c=y_kmeans1, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
pca_mouse

pca_mouse1 = pca_mouse.drop(['cluster'],axis=1)

pca_mouse1
from scipy.cluster.hierarchy import dendrogram, linkage

H_cluster = linkage(pca_mouse1,'ward')

plt.title('Hierarchical Clustering Dendrogram (truncated)')

plt.xlabel('sample index or (cluster size)')

plt.ylabel('distance')

dendrogram(

    H_cluster,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=9,  # show only the last p merged clusters

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,  # to get a distribution impression in truncated branches

)

plt.show()
# Assigning the clusters and plotting the observations as per hierarchical clustering

from scipy.cluster.hierarchy import fcluster

k=9

cluster_2 = fcluster(H_cluster, k, criterion='maxclust')

cluster_2[0:30:,]

plt.figure(figsize=(10, 8))

plt.scatter(pca_mouse1.iloc[:,0], pca_mouse1.iloc[:,1],c=cluster_2, cmap='prism')  # plot points with cluster dependent colors

plt.show()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsnes_results = tsne.fit_transform(m_scaled)

m_scaled['tsne-2d-one'] = tsnes_results[:,0]

m_scaled['tsne-2d-two'] = tsnes_results[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    palette=sns.color_palette("hls", 10),

    data=m_scaled,

    legend="full",

    alpha=0.3

)