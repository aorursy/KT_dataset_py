# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import cm

from sklearn.preprocessing import MaxAbsScaler

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



data = pd.read_csv('NetflixShows.csv',encoding='ISO-8859-1')



#NORMALIZAÇÃO:é pegar uma faixa de valores e levar ele para uma outra faixa de valores.

# Define os dados

dados = np.array(data[["release year", "ratingDescription"]])

# Instancia o MaxAbsScaler

p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador

p.fit(dados)

# Transforma os dados para ficar entre os valores -1 até 1

dados_normalizados = p.transform(dados)



#dados_normalizados[0:999]



#realização do K-means com os dados normalizados entre os valores de 1 a -1



X = dados_normalizados[0:999]

kmeans = KMeans(n_clusters = 3, init = 'random')

kmeans.fit(X)

kmeans.cluster_centers_

distancia = kmeans.fit_transform(X)

rotulos = kmeans.labels_



plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_),

plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centróides')