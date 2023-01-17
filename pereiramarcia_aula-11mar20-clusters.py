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
# carregando

df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')
df.head().T
# tamanhos e quantidades

df.info()
# encontrando o cliente sem limite de credito Nan

df[df['CREDIT_LIMIT'].isna()]
# encontrando os cliente com pagto minimo missing

df[df['MINIMUM_PAYMENTS'].isna()].sample(5).T
# preenchendo com zero os valores nulos

df.fillna(0,inplace=True)
#conferindo

df.info()
#k-means exige que todas as variáveis sejam numéricas



# excluir a coluna CUST_ID



df2 = df.drop('CUST_ID', axis=1)

df2.info()
# Importando o k-means



from sklearn.cluster import KMeans
# a identificação da soma dos erros quadráticos nos dá a quanridade ótima de clusters. ELBOW METODO para determinar 

# a quantidade ideal de cluster

# DETERMINAR A QUANTIDADE DE CLUSTER





# Elbow Method. Rodar a propria clusterização no dataset variando o valor de k (nro de cluster), calcular a soma quadrática

#dos erros (sse)



# o próprio k-mean já tem a propriedade de inertia que calcula o sse

sse= []

for k in range (1, 15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(df2)

    sse.append(kmeans.inertia_)
sse
import matplotlib.pyplot as plt



plt.plot(range(1, 15), sse, 'bx-')

plt.title('Elbow Method')

plt.xlabel('Numero de cluster')

plt.ylabel('SSE')

plt.show()
# vamos fazer com 8 clusters



kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)

cluster_id = kmeans.fit_predict(df2)
cluster_id
# vamos guardar os resultados como uma coluna no dataframe



df2['cluster_id'] = cluster_id



df2.sample(10).T
# Identificando as ocorrências do cluster 0



df2[df2['cluster_id']== 0].describe()
df2[df2['cluster_id']== 7].describe()
import seaborn as sns
sns.scatterplot(data=df2, x='BALANCE', y='PURCHASES', hue='cluster_id')