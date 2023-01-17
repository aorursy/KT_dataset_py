#Imporatando as bibliotecas

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas import DataFrame, Series

%matplotlib inline

import os

print(os.listdir("../input"))
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

pd.options.display.float_format = '{:,.2f}'.format
df_barragens = pd.read_csv('../input/database_versao_LatLongDecimal_fonteANM_23_01_2019.csv', header=0, delimiter=',',encoding = 'utf-8', decimal= ',')
df_barragens.head()
#Breve estatística descritiva

df_barragens.describe(include=['object'])
#Quantidade de barragem em cada estado

df_barragens['UF'].value_counts()[:5]
plt.figure(1 , figsize = (15 , 7))

# sns.countplot(x='UF', data=df_barragens, order=df_barragens["UF"].value_counts().index)

df_barragens['UF'].value_counts().plot(kind="barh")

plt.title('Número de barragens por estado')

plt.show()
#Quantidade de barragem em cada cidade de MG

df_barragens.where(df_barragens['UF'] == "MG")['MUNICIPIO'].value_counts()[:5]
plt.figure(1 , figsize = (15 , 7))

# sns.countplot(x='MUNICIPIO', data=df_cidades_mg, order=df_barragens["MUNICIPIO"].value_counts().index)

df_barragens.where(df_barragens['UF'] == "MG")['MUNICIPIO'].value_counts().plot(kind="bar")

plt.xticks(rotation = 90)

plt.title('Número de barragens por cidade em MG')

plt.show()
# Eliminando entradas que possuem valores nulos

df_barragens.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df_barragens['ALTURA_ATUAL_metros'] = df_barragens['ALTURA_ATUAL_metros'].apply(pd.to_numeric)

df_altura_estado = df_barragens.loc[df_barragens['UF'].isin(["MG", 'PA', 'SP', 'MT']) & df_barragens['ALTURA_ATUAL_metros']]

plt.figure(1 , figsize = (15 , 7))

sns.violinplot(x = 'UF' , y = 'ALTURA_ATUAL_metros' , data = df_altura_estado)

plt.title('Distribuicao da altura das barragens')

plt.xticks(rotation = 50)

plt.show()
df_barragens.dtypes