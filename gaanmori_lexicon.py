import io

import sys

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/lexiconemotions/lexicon_emotions_es.csv", sep=";")

#Convertir objetos (value) a flotante

data['value'] = data.value.str.replace(',', '.').astype(float)

data.info()

data.head()
#Calculo de la media

media = data.groupby([data['emotion']])['value'].mean()

media.index.names= ["Emociones"]

print("*************Media*************\n", media)
#Calcular la moda

moda = data.groupby([data['emotion']])['value'].agg(pd.Series.mode)

moda.index.names= ["Emociones"]

print("*************Moda*************\n", moda)
#Calculo de la desviacion estandar

desviacion = data.groupby([data['emotion']])['value'].std()

desviacion.index.names= ["Emociones"]

print("******Desviacion estandar******\n", desviacion)
#Valor maximo

maximo = data.groupby([data['emotion']])['value'].max()

maximo.index.names= ["Emociones"]

print("************Maximo************\n", maximo)

#Valor minimo

minimo = data.groupby([data['emotion']])['value'].min()

minimo.index.names= ["Emociones"]

print("************Minimo************\n", minimo)
sns.boxplot(x='emotion', y='value', data=data)

plt.show()
sns.FacetGrid(data, hue='emotion', size=5).map(sns.distplot, 'value').add_legend()
sns.pairplot(data, hue='emotion', height=5)

plt.show()
#Grafico de dispercion

sns.scatterplot(x='emotion',y='value',data=data)