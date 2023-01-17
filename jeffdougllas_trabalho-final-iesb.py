import matplotlib.pyplot as plt 

import pandas as pd 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

import numpy as np 

import seaborn as sns 

plt.style.use("seaborn-white")

import os

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import mean_squared_error

import matplotlib

from sklearn.metrics import accuracy_score
colors = [['#0D47A1','#1565C0','#1976D2','#1E88E5','#2196F3'],

          ['#311B92','#512DA8','#673AB7','#9575CD','#B39DDB'],

          ['#1B5E20','#388E3C','#4CAF50','#81C784','#66BB6A'],

          ['#E65100','#EF6C00','#F57C00','#FB8C00','#FF9800'],

          ['#3E2723','#4E342E','#5D4037','#6D4C41','#795548'],

          ['#BF360C','#D84315','#E64A19','#F4511E','#FF5722'],

          ['#880E4F','#AD1457','#C2185B','#D81B60','#E91E63']

         ]
train = '../input/train.csv'

test = '../input/test.csv'



df_train = pd.read_csv(train)

df_test = pd.read_csv(test)
df_train.head()
df_test.head()
#Dimensão datasets

print("Dimensão  dataset de treino")

print("Colunas:", df_train.shape[1],"\nlinhas:", df_train.shape[0])

print("-")

print("Dimensão  dataset de teste")

print("Colunas:", df_test.shape[1],"\nlinhas:", df_train.shape[0])
df_train.dtypes


df_test.dtypes
df_train.isnull().sum().sort_values(ascending=False)
#Preenchendo valores nulos

df_train.fillna(1, inplace=True)

df_test.fillna(1, inplace=True)
#Verificando os valores nulos no dataset de teste

df_test.isnull().sum().sort_values(ascending=False).head(10)
sp = df_train[df_train['estado'] == 'SP']

sp.groupby('municipio')['nota_mat'].mean().sort_values(ascending=False).head(10).plot(kind='barh',rot=0, figsize=(12,5),color=random.choice(colors), title="Top 10 - Maior nota média de matemática por município - SP")

plt.xlabel("Nota")

plt.ylabel("Município")

plt.show()
df_train.groupby(('municipio','estado', 'regiao'))['nota_mat'].mean().sort_values(ascending=True).head(10).plot(kind="barh",figsize=(12,5),color=random.choice(colors), title="Top 10 Menor nota média em matemática por município")

plt.xlabel("Nota")

plt.show()
#Listando novamente as colunas para validar a remoção

df_train.columns
#Agrupando Municipios e suas respectivas notas médias

df_train.groupby('municipio')['nota_mat'].mean()
#Calculando a simetria dos dados

#Um valor zero indica uma distribuição simétrica, um valor maior que zero ou menor indica uma distribuição assimétrica.

df_train.skew()
explode = (0.2, 0, 0, 0,0)

df_train.groupby('regiao')['populacao'].mean().plot(kind='pie',labeldistance=1.1, explode=explode,autopct='%1.0f%%', title="Percentual da população por região", shadow=True, startangle=90)

plt.ylabel(" ")

plt.show()
df_train.groupby('estado')['idhm'].mean().sort_values().plot(kind='barh', figsize=(15,8), title="IDH por estado", grid=True)

plt.xlabel("IDH-M")

plt.ylabel("ESTADO")

plt.show()


df_train.groupby(('municipio','estado', 'regiao'))['nota_mat'].mean().sort_values(ascending=False).head(20)
df_train.groupby('estado')['nota_mat'].describe()

df_train['nota_mat'].describe()
sns.distplot(df_train['nota_mat'].dropna(),kde=False,color='blue',bins = 20)

plt.show()
media_mt_estado = df_train.groupby('estado')['nota_mat'].max().sort_values()

media_mt_estado.plot(title = 'Maiores notas de matemática por estado', grid = False, kind='barh',color='red', figsize=(15,8))

plt.xlabel('Nota')

plt.ylabel('Estado')

plt.show()
df_train['nota_mat'] = np.log(df_train['nota_mat'])
df_train = df_train.append(df_test, sort=False)
df_train.shape
for c in df_train.columns:

    if (df_train[c].dtype == 'object') & (c != 'codigo_mun'):

        df_train[c] = df_train[c].astype('category').cat.codes
#Retirando a string ID_ID_ da Coluna

df_train['codigo_mun'] = df_train['codigo_mun'].str.replace('ID_ID_','')
df_train['nota_mat'].fillna(-2, inplace=True)
df_train.fillna(0, inplace=True)
df_train, df_test = df_train[df_train['nota_mat']!=-2], df_train[df_train['nota_mat']==-2]
df_train, valid = train_test_split(df_train, random_state=42)
rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df_train.columns if c not in ['nota_mat']]
df_train, df_test = df_train[df_train['nota_mat']!=-2], df_train[df_train['nota_mat']==-2]
rf.fit(df_train[feats], df_train['nota_mat'])
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
df_test[['codigo_mun','nota_mat']].to_csv('rf1.csv', index=False)
rf