# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregar o arquivo

df = pd.read_csv('../input/Mall_Customers.csv')



df.head()
# Verificar os dados



df.info()
# Descrevendo o dataframe

df.describe()
# Plotando os dados



df['Age'].plot.hist(bins=50)
df['Annual Income (k$)'].plot.hist(bins=50)
df['Gender'].value_counts().plot.bar()


import seaborn as sns
# Relação Sexo x Rendimentos

sns.stripplot(data=df, x='Gender', y='Annual Income (k$)', linewidth=1)
# Relação Score x Rendimento x Sexo

sns.scatterplot(data=df, x='Spending Score (1-100)',y='Annual Income (k$)',hue='Gender')
df.head(10)
# Relação Idade x Score x Sexo

sns.scatterplot(data=df, x='Age',y='Spending Score (1-100)',hue='Gender')
# Relação Idade x Rendimento x Sexo

sns.scatterplot(data=df, x='Age',y='Annual Income (k$)',hue='Gender')
# Importando a biblioteca

from sklearn.cluster import KMeans
df.columns
# Antes de começar vamos separar as colunas que vamos trabalhar

x1= df[['Spending Score (1-100)','Annual Income (k$)']]
# Precisamos determinar a quantidade de grupos



# Metodo Elbow (cotovelo)

# A idéia é rodar a própria clusterização k-means no dataset

# variando o valor de k (de 1 a 15 por exemplo) e para cada K

# vamos calcular a soma quadrática dos erros (sum of squared errors - sse)

sse = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(x1)

    sse.append(kmeans.inertia_)
# Verificando o resultado

sse
# Plotando os graficos

import matplotlib.pyplot as plt



plt.plot(range(1,15), sse, 'g^-')

plt.title('Método Elbow')

plt.xlabel('Número de clusters')

plt.ylabel('sse')

plt.show()
# Vamos executar a clusterização com 5 clusters

kmeans = KMeans(n_clusters=5, random_state=42)

cluster_id = kmeans.fit_predict(x1)
x1['cluster_id'] = cluster_id

x1.head(10)
# Plotando os agrupamentos e os centroides

fig = plt.figure(figsize=(12,6))



plt.scatter(x1['Spending Score (1-100)'],x1['Annual Income (k$)'],c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red', marker='x',s=200)

plt.title('Clusters Score x Rendimento')

plt.xlabel('Score')

plt.ylabel('Rendimento')

plt.show()
x2= df[['Age','Annual Income (k$)']]



sse = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(x2)

    sse.append(kmeans.inertia_)

 

# Plotando o cotovelo

plt.plot(range(1,15), sse, 'bs-')

plt.title('Método Elbow')

plt.xlabel('Número de clusters')

plt.ylabel('sse')

plt.show()
# Vamos executar a clusterização com 4 clusters

kmeans = KMeans(n_clusters=4, random_state=42)

cluster_id = kmeans.fit_predict(x2)



x2['cluster_id'] = cluster_id



# Plotando os agrupamentos e os centroides

fig = plt.figure(figsize=(12,6))



plt.scatter(x2['Age'],x2['Annual Income (k$)'],c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red', marker='x',s=200)

plt.title('Clusters Idade x Rendimento')

plt.xlabel('Idade')

plt.ylabel('Rendimento')

plt.show()