import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import os

#Carregando os dados

pasta_dados = os.listdir("../input")

df = pd.read_csv('../input/dados.csv')
#Conhecendo os dados 

df.head()
df_2 = pd.get_dummies(df)

df_2.head()
import seaborn as sns #visualização dos dados

plt.figure(figsize=(150, 100))

fig = sns.heatmap(df_2.corr(), annot=True)
dado = df_2[[ 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'contact_telephone', 'contact_cellular']].copy()
plt.figure(figsize=(25, 15))

fig = sns.heatmap(dado.corr(), annot=True)
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(dado, 'single')# chama objeto do algoritmo hierarquico

plt.figure(figsize=(10, 7))

dendrogram(linked, truncate_mode='lastp' ,orientation='top', distance_sort='descending', show_leaf_counts=True)# plot dedograma

plt.show()
from sklearn.cluster import  KMeans, DBSCAN, AgglomerativeClustering
kmeans = KMeans(n_clusters=2).fit(dado)

plt.hist(kmeans.labels_)

plt.show()
dbs = DBSCAN(eps=2).fit(dado) #Onde eps é a disntância máxima que um objeto pode estãr do outro

plt.hist(dbs.labels_)

plt.show()
agrupamento = AgglomerativeClustering( n_clusters=2, linkage='complete', affinity='l1').fit_predict(dado)

plt.hist(agrupamento)

plt.show()
from sklearn.metrics import silhouette_score

from sklearn.metrics.pairwise import euclidean_distances
matriz_distance = euclidean_distances(dado)

matriz_distance[0] #Um exemplo
# Utilizando o resultado do algoritmo Kmeans com k =2 

silhouette_score(matriz_distance, kmeans.labels_)
#Uilizando o resultado do algoritmo DBSCSN com eps = 2

silhouette_score(matriz_distance, dbs.labels_)
#Uilizando o resultado do algoritmo AgglomerativeClustering c

silhouette_score(matriz_distance, agrupamento)
work_data = df.copy() #cria uma copia dos dados

work_data['y'] = kmeans.labels_ #passa a classificação para seu respectivo objeto



acimda_de_60 = work_data[df['age'] > 60] #agora temos todos os objetos com mais de 60 anos



confiavel_60 = acimda_de_60['age'][acimda_de_60['y'] == 1]#objetos que são confiaveis 

nao_confiavel_60 = acimda_de_60['age'][acimda_de_60['y'] == 0] #objetos que não são confiaveis 
abaixo_de_60 = work_data[df['age'] < 60] #agora temos todos os objetos com mais de 60 anos



confiavel_menor_60 = abaixo_de_60['age'][abaixo_de_60['y'] == 1]#objetos que são confiaveis 

nao_confiavel_menor_60= abaixo_de_60['age'][abaixo_de_60['y'] == 0] #objetos que não são confiaveis 



fig, eixos = plt.subplots(nrows=1, ncols=2, figsize=(8,4))



pie_1 = eixos[0].pie([len(nao_confiavel_60), len(confiavel_60)], labels=['Não Confiavel ','Confiavel'],

                    autopct='%1.1f%%', colors=['gold', 'lightskyblue'])

# Define o título deste gráfico

eixos[0].set_title('Cima de 60 Anos')

# Deixa os dois eixos iguais, fazendo com que o gráfico mantenha-se redondo

eixos[0].axis('equal')

# Idem a acima, para o segundo gráfico de pizza

pie_2 = eixos[1].pie([len(nao_confiavel_menor_60), len(confiavel_menor_60)], labels=['Não Confiavel','Confiavel'], 

                    autopct='%1.1f%%', startangle=90, colors=['gold', 'lightskyblue'])

eixos[1].set_title('Abaixo de 60 Anos')

plt.axis('equal')

# Ajusta o espaço entre os dois gráficos

plt.subplots_adjust(wspace=1)

plt.show()