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
import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/test.csv')
df.head()
df.describe()
sns.pairplot(data=df)
df.plot.scatter(x='TP_ESCOLA', y='TP_COR_RACA')


df.TP_COR_RACA.plot.hist()



kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.TP_COR_RACA.values.reshape(-1,1))

labels = kmeans.predict(df.TP_COR_RACA.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfIdadeGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfIdadeGrupo 
cores = dfIdadeGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfIdadeGrupo.plot.scatter(x='TP_ESCOLA',y='TP_COR_RACA', c=cores)




kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df[['TP_ESCOLA','TP_COR_RACA']].values)

labels = kmeans.predict(df[['TP_ESCOLA','TP_COR_RACA']].values)

C = kmeans.cluster_centers_

print(labels,C)



dfGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfGrupo.sort_values(by='Grupo')





cores = dfGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfGrupo.plot.scatter(x='TP_ESCOLA',y='TP_COR_RACA', c=cores)



df.plot.scatter(x='NU_IDADE', y='NU_NOTA_REDACAO')
df.NU_NOTA_REDACAO.plot.hist()
df.NU_IDADE.plot.hist()
kmeans = KMeans(n_clusters=5)

kmeans = kmeans.fit(df.NU_IDADE.values.reshape(-1,1))

labels = kmeans.predict(df.NU_IDADE.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfGrupo.sort_values(by='Grupo')
cores = dfIdadeGrupo.Grupo.map({0:'b',1:'r',2:'y',3:'g',4:'p'})

dfIdadeGrupo.plot.scatter(x='NU_NOTA_REDACAO',y='NU_IDADE', c=cores)