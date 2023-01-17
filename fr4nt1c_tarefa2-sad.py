import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/test.csv")
df.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_numerico = df.drop(['NU_INSCRICAO','SG_UF_RESIDENCIA','TP_SEXO','CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','Q001','Q002','Q006','Q024','Q025','Q026','Q027','Q047'], axis=1)
imputer.fit(df_numerico)
df_no_null = imputer.transform(df_numerico)
df_tr = pd.DataFrame(df_no_null, columns=df_numerico.columns)

df_tr.isnull().sum()
df_tr.head()
df.describe()
sns.pairplot(data=df)
df_tr.plot.scatter(x='NU_IDADE', y='NU_NOTA_CN')

df_tr.plot.scatter(x='NU_IDADE', y='NU_NOTA_CH')

df_tr.plot.scatter(x='NU_IDADE', y='NU_NOTA_LC')
df_tr.plot.hist(y='NU_NOTA_CN')

df_tr.plot.hist(y='NU_IDADE')
df_tr.plot.hist(y='NU_NOTA_CH')

df_tr.plot.hist(y='NU_IDADE')
df_tr.plot.hist(y='NU_NOTA_LC')

df_tr.plot.hist(y='NU_IDADE')
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df_tr[['NU_IDADE','NU_NOTA_CN']].values)

labels = kmeans.predict(df_tr[['NU_IDADE','NU_NOTA_CN']].values)

C = kmeans.cluster_centers_

print(labels,C)



kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df_tr[['NU_IDADE','NU_NOTA_CH']].values)

labels = kmeans.predict(df_tr[['NU_IDADE','NU_NOTA_CH']].values)

C = kmeans.cluster_centers_

print(labels,C)



kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df_tr[['NU_IDADE','NU_NOTA_LC']].values)

labels = kmeans.predict(df_tr[['NU_IDADE','NU_NOTA_LC']].values)

C = kmeans.cluster_centers_

print(labels,C)
dfPetalGrupo = pd.concat([df_tr,pd.DataFrame(labels, columns = ['Grupo'])], axis=1, join='inner')
cores = dfPetalGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfPetalGrupo.plot.scatter(x='NU_IDADE',y='NU_NOTA_CN', c=cores)



cores = dfPetalGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfPetalGrupo.plot.scatter(x='NU_IDADE',y='NU_NOTA_CH', c=cores)



cores = dfPetalGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfPetalGrupo.plot.scatter(x='NU_IDADE',y='NU_NOTA_LC', c=cores)