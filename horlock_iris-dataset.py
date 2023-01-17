# Imports

import time

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.datasets import load_iris

%matplotlib inline



fontsize = 14

ticklabelsize = 14
# Carregando o dataset

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(len(df))

df.head()
# Imprima os valores numéricos da Variável target (o que queremos prever), 

# uma de 3 possíveis categorias de plantas: setosa, versicolor ou virginica

iris.target_names
# Imprima os valores numéricos da Variável target (o que queremos prever), 

# uma de 3 possíveis categorias de plantas: 0, 1 ou 2

iris.target
# Adicione ao dataset uma nova coluna com os nomes das espécies, pois é isso que vamos tentar prever (variável target

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()
# Inclua no dataset uma coluna com os valores numéricos da variável target

df['target'] = iris.target

df.head()
# Extraia as features (atributos) do dataset e imprima 

features = df.columns[:4]

features
# Calcule a média de cada feature para as 3 classes

df.groupby('target').mean().T
# Imprima uma Transposta do dataset (transforme linhas e colunas e colunas em linhas)

df.T.head(10)
# Utilize a função Info do dataset para obter um resumo sobre o dataset 

df.info()
# Faça um resumo estatístico do dataset

df.describe()
# Verifique se existem valores nulos no dataset

df.isnull().sum(axis=0)
# Faça uma contagem de valores de sepal length

df['sepal length (cm)'].value_counts(dropna=False)
# Crie um Histograma de sepal length

plt.hist(df['sepal length (cm)'])

plt.show()
# Crie um Gráficos de Dispersão (scatter Plot) da variável sepal length versus número da linha, 

# colorido por marcadores da variável target

plt.figure(figsize=(12, 8), dpi=80)

plt.scatter(range(len(df)), df['petal width (cm)'], c=df['target'])

plt.xlabel('Número da Linha', fontsize=fontsize)

plt.ylabel('Sepal length (cm)', fontsize=fontsize)

plt.title('Gráfico de Dispersão dos Atributos, colorido por marcadores da classe alvo', fontsize=fontsize)
# Crie um Scatter Plot de 2 Features (atributos)

plt.figure(figsize=(12, 8), dpi=80)

plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['target'])

plt.xlabel('petal length (cm)', fontsize=fontsize)

plt.ylabel('petal width (cm)', fontsize=fontsize)
# Crie um Scatter Matrix das Features (atributos)

attributes = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

pd.plotting.scatter_matrix(df[attributes], figsize=(16, 12))
# Crie um Histograma de todas as features

df.hist(figsize=(12,12))