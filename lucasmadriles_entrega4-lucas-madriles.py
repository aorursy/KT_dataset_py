# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/heart-disease-uci/heart.csv")

# Dataset com features que indicam condições de saúde e tem classe booliana de quadro de doença cardíaca

# será feito um algorítmo para prever se uma pessoa tem/terá doença cardíaca (1) ou não (0)
# separando em dois dataframes: features e classe

previsores = df.iloc[:, 0:13].values 

classe = df.iloc[:, 13].values
# Escalonamento: 

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

previsores = scaler.fit_transform(previsores)
# dividindo a base em treinamento e teste:

from sklearn.model_selection import train_test_split 

previsores_treinamento, previsores_teste, classe_treinamento,classe_teste = train_test_split(previsores,

                                                                                             classe,

                                                                                             test_size=0.3,

                                                                                             random_state=0)
# classificador:

from sklearn.svm import SVC

classificador = SVC(kernel='rbf',random_state=2, C=2.0)

# kernel = 'rbf'

classificador.fit(previsores_treinamento,classe_treinamento)
# classificação: 

previsoes = classificador.predict(previsores_teste)
# comparação com base de teste

from sklearn.metrics import confusion_matrix, accuracy_score 

precisao = accuracy_score(classe_teste, previsoes) # precisão do algoritmo

matriz = confusion_matrix(classe_teste, previsoes) # matriz de confusão 
precisao
matriz
import pandas as pd

df2 = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df2.head()
X = df2.iloc[:,1:5]
X
# transformando variável categórica em numérica em numerica:

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

X['Gender']=labelencoder.fit_transform(X['Gender'])
X = X.iloc[:,0:5].values
# padronização das variáveis

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
# cluster

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5)

kmeans.fit(X)
centroides = kmeans.cluster_centers_  # posição dos centroides

rotulos = kmeans.labels_ # grupo de cada registro
X = scaler.inverse_transform(X)
import matplotlib.pyplot as plt

cores = ["g.", "r.", "b.","k.","c."]  # lista de cores verde, vermelho e azul

for i in range(len(X)): # pra cada registro

    plt.plot(X[i][2], X[i][3], cores[rotulos[i]], markersize = 15) 

plt.scatter(centroides[:,0], centroides[:,1])  
cores = ["g.", "r.", "b.","k.","c."]  # lista de cores verde, vermelho e azul

for i in range(len(X)): # pra cada registro

    plt.plot(X[i][1], X[i][3], cores[rotulos[i]], markersize = 15) 

plt.scatter(centroides[:,0], centroides[:,1])  