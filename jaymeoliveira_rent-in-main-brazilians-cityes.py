import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
rent = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
rent['furniture'] = pd.get_dummies(rent.furniture)['furnished']
rent['animal'] = pd.get_dummies(rent.animal)['acept']

rent.columns = ['Cidade','Area', 'Quartos', 'Banheiros', 'Vagas', 'Andar', 'Permite Animal', 'Mobiliado', 'Condominio', 'Aluguel', 'IPTU', 'Seguro', 'Total']
rent['Tx. IPTU'] = 100*rent['IPTU']/rent['Aluguel']
rent['Tx. Seguro'] = 100*rent['Seguro']/rent['Aluguel']
rent
print('Tipo dos dados: ')
rent.info()
def muda_tipo(s):
    try:
        return int(s)
    except:
        return 0
rent.Andar = rent.Andar.map(muda_tipo)
plt.figure(figsize=(16, 12))
g = sns.FacetGrid(rent, row="Cidade", height=1.7, aspect=4,)
g.map(sns.distplot, "Area")
plt.figure(figsize=(16, 12))
g = sns.FacetGrid(rent, col="Cidade")
g.map(sns.countplot, "Mobiliado")
plt.figure(figsize=(16, 12))
g = sns.FacetGrid(rent, col="Cidade")
g.map(sns.countplot, "Permite Animal")
plt.figure(figsize=(16, 12))
g = sns.FacetGrid(rent, row="Cidade", height=1.7, aspect=4,)
g.map(sns.countplot, "Quartos")
rent.describe()
def remove_outliers(data, label):
    #calculate the IQR
    IQR = data[label].quantile(0.75) - data[label].quantile(0.25)
    
    #calculate the boundries
    lower = data[label].quantile(0.25) - (IQR * 1.5)
    upper = data[label].quantile(0.75) + (IQR * 1.5)
    
    # find the outliers
    outliers = np.where(data[label] > upper, True, np.where(data[label] < lower, True, False))
    
    # remove outliers from data.
    return data.loc[~outliers]
for label in ['Area', 'Aluguel', 'Quartos', 'Andar', 'IPTU','Condominio']:
    rent = remove_outliers(rent, label)
rent.describe()
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(16, 12))
sns.heatmap(rent.corr(), annot=True, fmt='.2f', vmax=0.8, vmin=-0.8)
plt.show()
sns.jointplot(y= rent['Aluguel'],x = rent['IPTU'],kind ='reg')
sns.jointplot(y= rent['Aluguel'],x = rent['Condominio'],kind ='reg')
sns.jointplot(y= rent['Aluguel'],x = rent['Seguro'],kind ='reg')
sns.jointplot(y= rent['Quartos'],x = rent['Area'],kind ='reg')
rent.query('Aluguel > 1500.0').sort_values('Aluguel')
rent.drop(columns=['Cidade','Permite Animal', 'IPTU', 'Seguro', 'Total'], inplace=True, errors='ignore')
rent.dropna(inplace=True)
rent
def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(dataset)
    s = metrics.silhouette_score(dataset, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(dataset, labels)
    calinski = metrics.calinski_harabasz_score(dataset, labels)
    return s, dbs, calinski, labels
s, dbs, calinski, labels = clustering_algorithm(3, rent)
print('Silhoutte: ', s, 'Davies-Bouldin: ', dbs, 'Calinski-Harabasz: ', calinski)
plt.figure(figsize=(16, 12))
plt.scatter(rent['Area'], rent['Aluguel'], c=labels, s=3, cmap='rainbow')
plt.xlabel("Area total")
plt.ylabel("Valor do aluguel")
plt.show()
def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(dataset)
    s = metrics.silhouette_score(dataset, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(dataset, labels)
    calinski = metrics.calinski_harabasz_score(dataset, labels)
    return s, dbs, calinski, labels
s, dbs, calinski, labels = clustering_algorithm(3, rent[rent.columns[1:]])
print('Silhoutte: ', s, 'Davies-Bouldin: ', dbs, 'Calinski-Harabasz: ', calinski)
unique, counts = np.unique(labels, return_counts=True)
dict(zip(unique, counts))
s2, dbs2, calinski2, labels2 = clustering_algorithm(5, rent[rent.columns[1:]])
print('Silhoutte: ', s2, 'Davies-Bouldin: ', dbs2, 'Calinski-Harabasz: ', calinski2)
unique, counts = np.unique(labels2, return_counts=True)
dict(zip(unique, counts))
plt.figure(figsize=(16, 12))
plt.scatter(rent['Andar'], rent['Aluguel'], c=labels, s=3, cmap='rainbow')
plt.xlabel("Area total")
plt.ylabel("Valor do aluguel")
plt.show()