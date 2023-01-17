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
# Carregadndo o arquivo



df = pd.read_csv('../input/Mall_Customers.csv')



df.head()
# verificando os dados



df.info() 
# Descrevendo os dados



df.describe()
# plotando os dados



df['Annual Income (k$)'].plot.hist(bins=30)
df['Gender'].value_counts().plot.bar()
#Visualizando os dados com a Seaborn

import seaborn as sns



# Relação Sexo x Rendimento

sns.stripplot(data=df, x='Gender', y = 'Annual Income (k$)')
#Realação Score x Rendimento x Sexo

sns.scatterplot(data=df, x='Spending Score (1-100)', y='Annual Income (k$)', hue='Gender')
#Realação Idade x Rendimento x Sexo

sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Gender')
#Realação Idade x Score x Sexo

sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender')
# importando o modelo



from sklearn.cluster import KMeans
# Antes de começar vamos separar as colunas que vamos trabalhar



X1 = df[['Spending Score (1-100)','Annual Income (k$)']]
# Precisamos determinar aquantidade de grupos



# Método Elbow

# A idéia é rodar a própria clusterização k-measn no dataset

# variando o valor de k ( de 1 a 15 por exemplo), e para cada k

# vamos calcular a soma quadrática dos erros (sum of squared errrs =sse)



sse = []





for k in range (1,15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(X1)

    sse.append(kmeans.inertia_)
# Verificando o resultado do sse

sse
# Plotanto o gráfico



import matplotlib.pyplot as plt



plt.plot(range(1,15), sse, 'bx-')



plt.title('Método Elbow')



plt.xlabel('Número de Clusters')

plt.ylabel('SSE')



plt.show()
# vamos executar a clusterização com 5 cluters



kmeans = KMeans(n_clusters=5, random_state=42)

cluster_id = kmeans.fit_predict(X1)
# vamos guardar os resultados no dataframe



X1['cluster_id']=cluster_id



X1.head(100)
# plotando os agrupamentos e os centroides



fig = plt.figure(figsize=(12,8))



plt.scatter(X1['Spending Score (1-100)'], X1['Annual Income (k$)'], c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red',marker='x',s=200)

plt.title('Clusters score x Rendimento')

plt.xlabel('Score')

plt.ylabel('Rendimento')

plt.show()
# Antes de começar vamos separar as colunas que vamos trabalhar



X2 = df[['Age','Annual Income (k$)']]
# Método Elbow



sse = []





for k in range (1,15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(X2)

    sse.append(kmeans.inertia_)

    

plt.plot(range(1,15), sse, 'bx-')



plt.title('Método Elbow')



plt.xlabel('Número de Clusters')

plt.ylabel('SSE')



plt.show()

# vamos executar a clusterização com 5 cluters



kmeans = KMeans(n_clusters=5, random_state=42)

cluster_id = kmeans.fit_predict(X2)



# vamos guardar os resultados no dataframe

X1['cluster_id']=cluster_id



# plotando os agrupamentos e os centroides



fig = plt.figure(figsize=(12,8))



plt.scatter(X2['Age'], X2['Annual Income (k$)'], c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red',marker='x',s=200)

plt.title('Clusters score x Rendimento')

plt.xlabel('Age')

plt.ylabel('Rendimento')

plt.show()