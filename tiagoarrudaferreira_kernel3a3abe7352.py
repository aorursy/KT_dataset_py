

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


df = pd.read_csv('../input/brasileiro-2018-2009/Brasileirao.csv')





df.head()



df.describe()
dfbrasileiro = df[['Idade_Media','Gols_feitos','Gols_sofridos','Ano','Posicao','Vitorias','Derrotas','Empates','Saldo','Qtd_Jogadores','Valor_total','Media_Valor']]



sns.pairplot(data=dfbrasileiro)


df.plot.scatter(x='Idade_Media', y='Valor_total')

df.Idade_Media.plot.hist()
df.Valor_total.plot.hist()
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Idade_Media.values.reshape(-1,1))

labels = kmeans.predict(df.Idade_Media.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfjogadorGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfjogadorGrupo
cores = dfjogadorGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfjogadorGrupo.plot.scatter(x='Idade_Media', y='Valor_total', c=cores)
dfjogador = df[['Gols_feitos','Gols_sofridos','Estrangeiros','Posicao','Idade_Media','Vitorias','Derrotas','Empates','Saldo','Qtd_Jogadores','Valor_total','Media_Valor']]

dfjogador.head()




sns.pairplot(data=dfjogador, kind="reg")



dfjogador = df[['Posicao','Vitorias']]

dfjogador.head()
sns.pairplot(data=dfjogador, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()




X = np.array(dfjogador['Vitorias']).reshape(-1, 1)

y = le.fit_transform(dfjogador['Posicao'])

regressao.fit(X, y)







V = 20

print('Posição: ',regressao.predict(np.array(V).reshape(-1, 1)))


