# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carregando o arquivo

df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')



df.shape
df.head().T
# Tipos e quantidades

df.info()
# Vamos descobrir o fdp com limite de crédito igual a Nan

df[df['CREDIT_LIMIT'].isna()]

#Preenchendo os valores Nan com 0

df.fillna(0, inplace=True)
df.info()
# estatística descritiva primária

df.describe().T
# para fazer a clusterização só podemos ter valores numéricos

# e o CUST_ID (customer id) não serve para segregar os clientes

df2 = df.drop('CUST_ID', axis=1)



df2.info()
#Yellowbrick - dar olhada - https://www.scikit-yb.org/en/latest/api/cluster/elbow.html



# Elbow Method

# A ideia é rodar a própria clusterização (k-means) no dataset

# variando o valor de k (de 1 a 15, por exemplo)

# e para cada valor de k vamos calcular a

# soma quadrática dos erros (sum of squared error - sse)

# Importando o modelo

from sklearn.cluster import KMeans



# Lista de sse

sse = []



for k in range(1, 15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(df2)

    sse.append(kmeans.inertia_)



# Verificando see

sse
# Plotando o gráfico de cotovelo

import matplotlib.pyplot as plt



plt.plot(range(1, 15), sse, 'bx-')

plt.title('Elbow Method')

plt.xlabel('Número de clusters')

plt.ylabel('SSE')

plt.show()
# Vamos excutar a clusterização com k=8

# init{‘k-means++’, ‘random’} 

# or ndarray of shape (n_clusters, n_features), default=’k-means++’

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)

cluster_id = kmeans.fit_predict(df2)
# Vamos olhar o cluster_id

cluster_id
# Vamos guardar os resultados no dataframe

df2['cluster_id'] = cluster_id



df2.sample(5).T
df2 = df2.drop('luster_id', axis=1)
df2.sample(5).T
# São muitas colunas e isso dificulta visualizar os cluster e entendê-los

import seaborn as sns

sns.pairplot(df2, hue='cluster_id')
# relacionando purchases e credit_limit

sns.scatterplot(data=df2, x='PURCHASES', y='CREDIT_LIMIT', hue=cluster_id)
# descrevendo o cluster 0

df2[df2['cluster_id'] == 2].describe().T