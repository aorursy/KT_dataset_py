# Import delle l'analisi esplorativa dei dati
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

# import delle librerie richieste per l'applicazione di algoritmi di clustering
import sklearn
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
store = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
store.head()
store.describe()
store.shape
store.info()
store.describe()
#Set dello stile dei grafici
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (10 , 5))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(store[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()
plt.figure(1 , figsize = (10 , 5))
sns.countplot(y = 'Gender' , data = store)
plt.show()
plt.figure(1 , figsize = (10 , 5))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = store)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()
plt.figure(1 , figsize = (10 , 5))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = store[store['Gender'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income suddiviso per Gender')
plt.legend()
plt.show()
plt.figure(1 , figsize = (10 , 5))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = store[store['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score suddiviso per Gender')
plt.legend()
plt.show()
plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'Gender' , data = store , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'Gender' , data = store)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()
# Metodo k-Means con un numero di clusters arbitrario.
# - n_clusters: numero di cluster desiderati - limitazione del k-Means;
# - n_init: esegue l'algoritmo n volte in modo indipendente, con diversi centroidi casuali per scegliere il modello finale come quello con il SSE più basso.
# - max_iter: indica il numero massimo di iterazioni per ogni singola esecuzione. 

X1 = store[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    method = (KMeans(n_clusters = n, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 1))
    method.fit(X1)
    inertia.append(method.inertia_)
#Stampa delle etichette relative ai cluster
method.labels_
# Plot del valore della somma della radice delle distanze al crescere del numero dei cluster

plt.figure(1 , figsize = (10 ,5))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Numero dei cluster') , plt.ylabel('Sum of Squared Distance')
plt.show()
method = (KMeans(n_clusters = 4, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 1))
method.fit(X1)
labels1 = method.labels_
centroids1 = method.cluster_centers_
#Stampa delle etichette predette
method.labels_
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = method.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (10 , 5) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age', y = 'Spending Score (1-100)', data = store, c = labels1, s = 50 )
plt.scatter(x = centroids1[: , 0], y =  centroids1[: , 1], s = 50, c = 'red', alpha = 0.5)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Age')
plt.show()
X2 = store[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    method = (KMeans(n_clusters = n, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 1))
    method.fit(X2)
    inertia.append(method.inertia_)
# Plot del valore della somma della radice delle distanze al crescere del numero dei cluster

plt.figure(1 , figsize = (10 ,5))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Numero dei cluster') , plt.ylabel('Sum of Squared Distance')
plt.show()
method = (KMeans(n_clusters = 5, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 1))
method.fit(X2)
labels2 = method.labels_
centroids2 = method.cluster_centers_
#Stampa delle etichette predette
method.labels_
h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = method.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1, figsize = (10 , 6))
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = store , c = labels2, s = 50)
plt.scatter(x = centroids2[: , 0], y =  centroids2[: , 1], s = 50, c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Annual Income (k$)')
plt.show()
X3 = store[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    method = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, tol=0.0001,  random_state= 1))
    method.fit(X3)
    inertia.append(method.inertia_)
# Plot del valore della somma della radice delle distanze al crescere del numero dei cluster

plt.figure(1 , figsize = (10 ,5))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Numero dei cluster') , plt.ylabel('Sum of Squared Distance')
plt.show()
# Definizione della lista del numero di cluster da testare
range_n_clusters = list(x for x in range (2,10+1))

for num_clusters in range_n_clusters:
    method = KMeans(n_clusters = num_clusters ,init='k-means++', n_init = 10 ,max_iter=300, tol=0.0001,  random_state= 1)
    method.fit(X3)
    cluster_labels = method.labels_
    # Calcolo coefficiente di silhouette
    silhouette_avg = silhouette_score(X3, cluster_labels)
    print("Per n_clusters={0}, il coefficiente di Silhouette è pari a {1}".format(num_clusters, silhouette_avg))
method = (KMeans(n_clusters = 6, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 1))
method.fit(X3)
labels3 = method.labels_
centroids3 = method.cluster_centers_
#Stampa delle etichette predette
method.labels_
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
store['Cluster_Id'] = method.labels_
# Stampa dei primi 5 esempi
store.head()
features_list = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in features_list:
    sns.boxplot(x='Cluster_Id', y=feature, data=store)
    plt.show()
# Applicazione del metodo Single Linkage
plt.figure(figsize = (10,5))
single_linkage = linkage(X3, method="single", metric='euclidean')
dendrogram(single_linkage)
plt.show()
# Applicazione del metodo Complete linkage
plt.figure(figsize = (10,5))
complete_linkage = linkage(X3, method="complete", metric='euclidean')
dendrogram(complete_linkage)
plt.show()
# Applicazione del metodo Average linkage
plt.figure(figsize = (10,5))
avg_linkage = linkage(X3, method="average", metric='euclidean')
dendrogram(avg_linkage)
plt.show()
# Desiderando un numero di cluster pari a 4, si inizializza il parametro n_clusters=4
cluster_labels = cut_tree(complete_linkage, n_clusters=4).reshape(-1, )
#Stampa delle etichette dei cluster
cluster_labels
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
store['Cluster_Labels'] = cluster_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
store.head()
#Plot delle Features

features_list = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in features_list:
    sns.boxplot(x='Cluster_Labels', y=feature, data=store)
    plt.show()
## Numero dei clienti in ciascun cluster
store['Cluster_Labels'].value_counts(ascending=True)
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
agglomerative_cluster_labels = ac.fit_predict(X3)
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
store['Agglomerative_Clustering'] = agglomerative_cluster_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
store.head()
#Plot delle Features

features_list = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in features_list:
    sns.boxplot(x='Agglomerative_Clustering', y=feature, data=store)
    plt.show()
from sklearn.cluster import DBSCAN
X3 = StandardScaler().fit_transform(X3)

dbscan = DBSCAN(eps=0.3, min_samples=5, metric = 'euclidean')
dbscan.fit(X3)
dbscan_labels = dbscan.labels_
#Stampa delle etichette dei cluster
dbscan_labels
#Identificazione numero di cluster e punti rumorosi
n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_ = list(dbscan_labels).count(-1)

print('Numero di cluster stimati: %d' % n_clusters_)
print('Numero di punti rumorosi identificati: %d' % n_noise_)
# Assegnazione delle etichette a ciascun esempio presente nel DataFrame
store['DensityBased_Labels'] = dbscan_labels
# Stampa dei primi 5 elementi presenti nel DataFrame
store.head()
#Plot delle Features

features_list = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in features_list:
    sns.boxplot(x='DensityBased_Labels', y=feature, data=store)
    plt.show()
