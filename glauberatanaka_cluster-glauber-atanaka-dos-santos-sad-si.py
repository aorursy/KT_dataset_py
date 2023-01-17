import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()
df.describe()
df.plot.scatter(x='Age',y='Annual Income (k$)')
df.Age.plot.hist()
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Age.values.reshape(-1,1))

labels = kmeans.predict(df.Age.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfIdadeGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfIdadeGrupo 
cores = dfIdadeGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfIdadeGrupo.plot.scatter(x='Age',y='Annual Income (k$)', c=cores)