import numpy as np                

import pandas as pd               

import seaborn as sns             

import matplotlib.pyplot as plt   

import scipy.stats                

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

import time

import warnings

from sklearn import cluster, datasets

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

import os
lista = os.listdir("../input")

print(lista)
df = pd.read_csv("../input/"+lista[1])

df.head(5)
lista2 = [x.replace(".csv","") for x in lista ]

close = pd.DataFrame([0.0]*len(pd.read_csv("../input/"+lista[1])))

vol = pd.DataFrame([0.0]*len(pd.read_csv("../input/"+lista[1])))

change = pd.DataFrame([0.0]*len(pd.read_csv("../input/"+lista[1])))



for i in range(len(lista)):

    a = pd.read_csv("../input/"+lista[i])

    fecha = pd.DataFrame({"date":a.date})    

    close[lista2[i]] = a.close

    vol[lista2[i]] = a.volume

    change[lista2[i]] = a.change

print(close.shape == vol.shape == change.shape)

print(close.shape)
print("Se tienen", len(lista), "acciones")
close = close.iloc[:,1:31]

vol = vol.iloc[:,1:31]

change = change.iloc[:,1:31]
close.head(5)
sns.clustermap(close.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(13, 13))
sns.clustermap(vol.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(13, 13))
sns.clustermap(change.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(13, 13))
df_scale = change.copy()

scaler = preprocessing.StandardScaler()

columns =change.columns

df_scale[columns] = scaler.fit_transform(df_scale[columns])

df_scale.head()
compl = pd.DataFrame({'Porcentaje de Completitud':df_scale.count()*100/len(df_scale)}).round(2)

compl['indexx'] = compl.index.values.tolist()

compl = compl.sort_values(['Porcentaje de Completitud'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(15,15))

sns.barplot(x=compl["Porcentaje de Completitud"],y=compl.indexx,palette='Blues_d')

plt.axvline(x=compl.mean()[0],linestyle='--',color='firebrick',label='Completitud Promedio: Acciones DJI')

plt.xlabel("Porcentaje de Completitud",fontsize=15)

plt.ylabel("Variable",fontsize=15)

plt.title("Completitud Promedio: Acciones DJI",fontsize=15)

plt.show()
a = pd.DataFrame(df_scale.describe()).mean(axis=1)[1]

print('La media de los retornos de todas las acciones es:', a)
df_scale = df_scale.fillna(a)

df_scale.shape
#Elbow graph

ks = range(1, 6)

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(df_scale.iloc[:,1:])

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
model = KMeans(n_clusters=3)



# Fit model to points

model.fit(df_scale)



# Determine the cluster labels of new_points: labels

df_scale['cluster'] = model.predict(df_scale)



df_scale.head()
# Create PCA instance: model

model_pca = PCA()



# Apply the fit_transform method of model to grains: pca_features

pca_features = model_pca.fit_transform(df_scale)



# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = pca_features[:,1]



# Scatter plot xs vs ys

sns.scatterplot(x=xs, y=ys, hue="cluster", data=df_scale)
centroids = model.cluster_centers_

df_scale.groupby(['cluster']).mean()
cero = []

uno = []

dos = []



cero = (df_scale.cluster==0).astype(int)*df_scale.AAPL

uno = (df_scale.cluster==1).astype(int)*df_scale.AAPL

dos = (df_scale.cluster==2).astype(int)*df_scale.AAPL



for i in range(len(cero)):

    if(cero[i] == 0):

        cero[i] = np.nan

    if(uno[i] == 0):

        uno[i] = np.nan

    if(dos[i] == 0):

        dos[i] = np.nan
conf = df_scale.cluster

sns.set(style ="dark")

ax = sns.countplot(x = conf, palette='Blues_d')

ax.set_title(label='Frecuencia', fontsize=20)
sns.heatmap(df_scale.groupby(['cluster']).mean(), cmap="vlag")
plt.figure(figsize=(20,7))

plt.title('Retornos de APPL según el Regímen de Mercado')

plt.plot(cero,color='g',label="0")

plt.plot(uno,color='r',label="1")

plt.plot(dos, color = 'b',label="2")

plt.legend()

plt.xticks([])

plt.yticks([])

plt.ylim((-6, 6))
df2 = change.copy()

scaler = preprocessing.StandardScaler()

columns =change.columns

df2[columns] = scaler.fit_transform(df2[columns])

df2.head()
a = pd.DataFrame(df2.describe()).mean(axis=1)[1]

df2 = df2.fillna(a)

df2 = df2.values

print(df2)
print(df2.shape)
X = df2

plt.figure(figsize=(20,10))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))

plt.title('Dendograma')

plt.show()
hc = AgglomerativeClustering(n_clusters = 3, 

                    affinity = 'euclidean', 

                    linkage = 'ward')



y_hc = hc.fit_predict(X)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 10, c = 'green', label = 'Cluster 3')

plt.legend()

plt.show()
clusters = pd.DataFrame({'cluster':y_hc})

clusters = clusters.cluster.values.tolist()

conf = clusters

sns.set(style ="dark")

ax = sns.countplot(x = conf, palette='Blues_d')

ax.set_title(label='Frecuencia', fontsize=20)
df_scale['cluster2'] = clusters

sns.heatmap(df_scale.groupby(['cluster2']).mean(), cmap="vlag")

cero = []

uno = []

dos = []



cero = (df_scale.cluster2==0).astype(int)*df_scale.AAPL

uno = (df_scale.cluster2==1).astype(int)*df_scale.AAPL

dos = (df_scale.cluster2==2).astype(int)*df_scale.AAPL



for i in range(len(cero)):

    if(cero[i] == 0):

        cero[i] = np.nan

    if(uno[i] == 0):

        uno[i] = np.nan

    if(dos[i] == 0):

        dos[i] = np.nan
plt.figure(figsize=(20,7))

plt.title('Retornos de APPL según el Regímen de Mercado: Clustering Jerárquico')

plt.plot(cero,color='g',label="0")

plt.plot(uno,color='r',label="1")

plt.plot(dos, color = 'b',label="2")

plt.legend()

plt.xticks([])

plt.yticks([])

plt.ylim((-6, 6))
df_scale['DJI'] = df_scale.iloc[:,:30].sum(axis=1)
cero = []

uno = []

dos = []



cero = (df_scale.cluster==0).astype(int)*df_scale.DJI

uno = (df_scale.cluster==1).astype(int)*df_scale.DJI

dos = (df_scale.cluster==2).astype(int)*df_scale.DJI



for i in range(len(cero)):

    if(cero[i] == 0):

        cero[i] = np.nan

    if(uno[i] == 0):

        uno[i] = np.nan

    if(dos[i] == 0):

        dos[i] = np.nan

        

        

cero1 = []

uno1 = []

dos1 = []



cero1 = (df_scale.cluster2==0).astype(int)*df_scale.DJI

uno1 = (df_scale.cluster2==1).astype(int)*df_scale.DJI

dos1 = (df_scale.cluster2==2).astype(int)*df_scale.DJI



for i in range(len(cero1)):

    if(cero1[i] == 0):

        cero1[i] = np.nan

    if(uno1[i] == 0):

        uno1[i] = np.nan

    if(dos1[i] == 0):

        dos1[i] = np.nan
plt.figure(figsize=(20,7))

plt.title('Retornos de DJI según el Regímen de Mercado: Clustering K-Means')

plt.plot(cero,color='g',label="0")

plt.plot(uno,color='r',label="1")

plt.plot(dos, color = 'b',label="2")

plt.legend()

plt.xticks([])

plt.yticks([])

plt.figure(figsize=(20,7))

plt.title('Retornos de DJI según el Regímen de Mercado: Clustering Jerárquico')

plt.plot(cero1,color='g',label="0")

plt.plot(uno1,color='r',label="1")

plt.plot(dos1, color = 'b',label="2")

plt.legend()

plt.xticks([])

plt.yticks([])
bla = df2

plt.figure(figsize=(10,10))

plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,

                    hspace=.01)

plot_num = 1



default_base = {'n_neighbors': 10,

                'n_clusters': 3}

datasets = [

    (bla, {'n_clusters': 2}),(bla, {'n_clusters': 3}),(bla, {'n_clusters': 4})]



for i_dataset, (dataset, algo_params) in enumerate(datasets):

    params = default_base.copy()

    params.update(algo_params)

    X, y = dataset, dataset

    X = StandardScaler().fit_transform(X)

    ward = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='ward')

    complete = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='complete')

    average = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='average')

    single = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='single')

    clustering_algorithms = (

        ('Single Linkage', single),

        ('Average Linkage', average),

        ('Complete Linkage', complete),

        ('Ward Linkage', ward),

    )

    for name, algorithm in clustering_algorithms:

        t0 = time.time() 

        with warnings.catch_warnings():

            warnings.filterwarnings(

                "ignore",

                message="the number of connected components of the " +

                "connectivity matrix is [0-9]{1,2}" +

                " > 1. Completing it to avoid stopping the tree early.",

                category=UserWarning)

            algorithm.fit(X)

        t1 = time.time()

        if hasattr(algorithm, 'labels_'):

            y_pred = algorithm.labels_.astype(np.int)

        else:

            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)

        if i_dataset == 0:

            plt.title(name, size=9)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',

                                             '#f781bf', '#a65628', '#984ea3',

                                             '#999999', '#e41a1c', '#dede00']),

                                      int(max(y_pred) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)

        plt.ylim(-2.5, 2.5)

        plt.xticks(())

        plt.yticks(())

        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

                 transform=plt.gca().transAxes, size=15,

                 horizontalalignment='right')

        plot_num += 1

plt.show()