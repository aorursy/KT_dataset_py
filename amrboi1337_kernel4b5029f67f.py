# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
# Import das bibliotecas necessárias para rodar o programa

from sklearn.preprocessing import StandardScaler

import seaborn as sns

from sklearn.cluster import KMeans
# Lendo os dados do Dataset

dataset = pd.read_csv('../input/CC GENERAL.csv')

print(dataset.shape)
# Visualizando as primeiras 5 instâncias do Dataset

dataset.head()
# Analisando mais detalhadamente cada coluna do Dataset

dataset.describe()
# Verificando o tipo dos dados presentes na tabela

dataset.dtypes
dataset.isnull().sum()
# Utilizando a moda dos dados da coluna CREDIT_LIMIT

moda_credit_limit = dataset.CREDIT_LIMIT.mode()[0]

moda_minimum_payments = dataset.MINIMUM_PAYMENTS.mode()[0]



moda_credit_limit, moda_minimum_payments
# Substituindo valores categóricos pela moda dos valores não nulos presentes na coluna CREDIT_LIMIT

dataset['CREDIT_LIMIT'].fillna(moda_credit_limit, inplace = True)



# Substituindo valores categóricos pela moda dos valores não nulos presentes na coluna MINIMUM_PAYMENTS

dataset['MINIMUM_PAYMENTS'].fillna(moda_minimum_payments, inplace = True)



dataset.isnull().sum().any()
# Desconsiderando a coluna do identificador CUST_ID 

dataset.drop(['CUST_ID'], axis= 1, inplace = True)



# Nenhum valor categórico encontrado

X = dataset.iloc[:,:].values
# Analisando a correlação dos dados do Dataset

# Para isso, plotamos a matriz de correlação entre os valores numéricos



rain_data_num = dataset[['BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE',

                       'PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',

                      'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS',

                      'MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]

plt.figure(figsize=(12,8))

sns.heatmap(rain_data_num.corr(),annot=True,cmap='bone',linewidths=0.25)
# Desconsiderando a coluna CASH_ADVANCE_TRX

dataset.drop(columns=['CASH_ADVANCE_TRX', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES'])

dataset.columns
standardscaler = StandardScaler()

X = standardscaler.fit_transform(X)
clusters = []

for i in range(1, 11):

    kmeans= KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    # inertia_ : Soma das distâncias quadradas das amostras para o centroide mais próximo

    clusters.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize=(12, 8))

sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)

ax.set_title('Procurando o Valor de K "Joelho"')

ax.set_xlabel('Clusters')

ax.set_ylabel('Inertia')



# Annotate arrow

ax.annotate('Possível "Joelho"', xy=(4, 100000), xytext=(4, 80000), xycoords='data',          

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2))



ax.annotate('Possível "Joelho"', xy=(6, 85000), xytext=(6, 100000), xycoords='data',          

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2))



plt.show()
# Escolhendo o número de clusters K = 4 a partir da visualização do gráfico

kmeans4 = KMeans(n_clusters=4).fit(X)
# Plotando o gráfico de agrupamento dos dados com K = 4

# BALANCE: Valor do saldo deixado nas contas deles para fazer compras

# PURCHASES_TRX : Número de transações de compras feitas 

plt.figure(figsize=(12, 8)) 

sns.scatterplot(X[:,0], X[:,11], hue=kmeans4.labels_, 

                palette=sns.color_palette('hls', 4))



plt.title('KMeans with 4 Clusters')

plt.xlabel('BALANCE')

plt.ylabel('PURCHASES_TRX')

plt.show()
# Escolhendo o número de clusters K = 6 a partir da visualização do gráfico

kmeans6 = KMeans(n_clusters=6).fit(X)
# Plotando o gráfico de agrupamento dos dados com K = 6

plt.figure(figsize=(12, 8))

sns.scatterplot(X[:,0], X[:,11], hue=kmeans6.labels_, 

                palette=sns.color_palette('hls', 6))



plt.title('KMeans with 6 Clusters')

plt.xlabel('BALANCE')

plt.ylabel('PURCHASES_TRX')

plt.show()
# 4 grupos

labels=kmeans4.labels_

clusters=pd.concat([dataset, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()



for c in clusters:

    grid= sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)
# 6 grupos

labels=kmeans6.labels_

clusters=pd.concat([dataset, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()



for c in clusters:

    grid= sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)