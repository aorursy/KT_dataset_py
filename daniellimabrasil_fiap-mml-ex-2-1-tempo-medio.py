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
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
df = pd.read_csv('../input/tempomedio/Usuarios-TempoMedioQuantidade.csv', sep=';',encoding = "ISO-8859-1", engine='python')
df.head()
samples = df[['Quantidade', 'TempoMedio']]
logins = df[['Login']]
# samples_d = pd.get_dummies(df, prefix=['Dum'], columns=['NomeEquipe'])
samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)
samples_scaled
#### Solução com KMEANS ####
model = KMeans(n_clusters=4)
model.fit(samples_scaled)
#Imprime os centros dos clusters
print(model.cluster_centers_)

print(model.labels_)
print(set(model.labels_))
#Método Elbow para encontrar o melhor valor de k
inertias =[]
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i).fit(samples_scaled)    
    #somatório dos erros quadráticos das instâncias de cada cluster
    inertias.append(kmeans.inertia_)
    
plt.figure(1)
plt.plot(range(1, 15), inertias)
plt.title('O Metodo Elbow')
plt.xlabel('No de clusters')
plt.ylabel('WSS - within cluster sum of squares')
plt.show()
#O melhor k é igual a 4 (logo 4 clusters)
kmeans4 =KMeans(n_clusters=4)
kmeans4.fit(samples_scaled)
df['cluster'] = kmeans4.labels_
df.head()
xs = df[['Quantidade']].values
ys = df[['TempoMedio']].values
plt.xlabel("Quantidade")
plt.ylabel("Tempo Médio")
cluster = df[['cluster']].values
plt.scatter(xs, ys, c=cluster)
plt.show()
#### Agora solução com DBSCAN ####
from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps=0.1)
clusters = dbscan.fit_predict(samples_scaled)
# plot the cluster assignments
plt.scatter(samples_scaled[:, 0], samples_scaled[:, 1], c=clusters, cmap="plasma")
plt.xlabel("Quantidade")
plt.ylabel("Tempo Médio")
### NO DBSCAN, POR NÃO LIDAR BEM COM OUTLIERS (A MEU VER), NÃO CONSEGUIU AGRUPAR CORRETAMENTE ###
##
## Código do Professor ##

from sklearn.neighbors import NearestNeighbors

#Selecionando o melhor valor para o eps
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

db = DBSCAN(eps=0.12, min_samples=5).fit(X)
labels = db.labels_
print(set(labels))