# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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
df = pd.read_csv('../input/cidadedigital-br/Cidades-Digitais-t.csv')
df.head()
df.describe()
sns.pairplot(data=df)
df.plot.scatter(x='Valor_Previsto', y='Pontos_Atendidos')
df.describe()
df.Valor_Previsto.plot.hist()
df.Pontos_Atendidos.plot.hist()


kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Valor_Previsto.values.reshape(-1,1))

labels = kmeans.predict(df.Valor_Previsto.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)



dfAtendidoGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfAtendidoGrupo 
cores = dfAtendidoGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfAtendidoGrupo.plot.scatter(x='Valor_Previsto',y='Pontos_Atendidos', c=cores)
dfvalorpopulacao = df[['UF','População_Estimada_(2016)','Valor_Pago_(Bruto)','Pontos_Atendidos','Valor_Previsto']]

dfvalorpopulacao.head()
sns.pairplot(data=dfvalorpopulacao, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()
X = np.array(dfvalorpopulacao['Valor_Pago_(Bruto)']).reshape(-1, 1)

y = le.fit_transform(dfvalorpopulacao['Pontos_Atendidos'])

regressao.fit(X, y)
tamanho = 40223.00

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))