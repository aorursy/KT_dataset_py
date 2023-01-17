import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings



df = pd.read_csv(r'../input/Mall_Customers.csv')

df.head()
plt.figure(1 , figsize = (16 , 16))

n = 0

for x in ['Age' , 'Annual Income (k$)' , 'Gender']:

    n += 1

    plt.subplot(3 , 1 , n)

    sns.boxplot(x=x, y='Spending Score (1-100)', data=df, whis='range')

    plt.title('Distplot of {}'.format(x))
df_p = df.drop(['CustomerID'], axis=1)

df_p['Gender'] = df_p['Gender'].factorize()[0]
from sklearn.manifold import TSNE

tsn = TSNE()

res_tsne = tsn.fit_transform(df_p)
from sklearn.cluster import AgglomerativeClustering as AggClus

clus_mod = AggClus(n_clusters = 5) # no reason, just a try

assign = clus_mod.fit_predict(df_p)

plt.figure(figsize=(8,8))

sns.set(style='whitegrid',palette='pastel')

cmap = sns.cubehelix_palette(dark=.5, light=.5, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='Paired');
from scipy.cluster.hierarchy import dendrogram, ward

sns.set(style='whitegrid')

plt.figure(figsize=(8,8))

link = ward(res_tsne)

dendrogram(link)

ax = plt.gca()

bounds = ax.get_xbound()

ax.plot(bounds, [30,30],'--', c='m')

ax.plot(bounds,'--', c='c')
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 7):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 6 ,max_iter=300, 

                        tol=0.0001,  random_state= 101  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (16 ,6))

plt.plot(np.arange(1 , 7) , inertia , 'o')

plt.plot(np.arange(1 , 7) , inertia , '-' )

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 6 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 



plt.figure(1 , figsize = (16 , 6) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.hsv, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 200 , c = 'w' )

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 7):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 6 ,max_iter=300, 

                        tol=0.0001,  random_state= 101  , algorithm='elkan') )

    algorithm.fit(X2)

    inertia.append(algorithm.inertia_)

                   

    

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 6 ,max_iter=300, 

                        tol=0.0001,  random_state= 101  , algorithm='elkan') )

algorithm.fit(X2)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_





h = 0.02

x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1

y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (16 , 6) )

plt.clf()

Z2 = Z2.reshape(xx.shape)

plt.imshow(Z2 , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Set3, aspect = 'auto', origin='lower')



plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 

            s = 200 )

plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 200 , c = 'w' )

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')