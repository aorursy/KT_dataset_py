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
# Carregando os dados

df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='latin')
# Verificando o tamanho do data frame

df.shape
# Verificando os metadados do data frame

df.info()
# Exibindo o nome das colunas

df.columns
# Exibindo o Data Frame

df.head()
# Renomeando e filtrando apenas as colunas necessárias para o estudo

df.rename(columns={'iyear':'Ano','imonth':'Mês','iday':'Dia','country_txt':'País',

                   'region_txt':'Região','city':'Cidade','latitude':'Latitude',

                   'longitude':'Longitude','attacktype1_txt':'Tipo de Ataque','target1':'Alvo',

                   'nkill':'Número de Mortos','nwound':'Número de Feridos','gname':'Grupo',

                   'targtype1_txt':'Tipo de Alvo','weaptype1_txt':'Tipo de Arma','motive':'Motivo'},

          inplace=True)



df = df[['Ano','Mês','Dia','País','Região','Cidade','Latitude','Longitude','Tipo de Ataque','Alvo',

         'Número de Mortos','Número de Feridos','Grupo','Tipo de Alvo','Tipo de Arma','Motivo']]



df.head()
# Verificando os metadados do data frame, somente com as colunas selecionadas

df.info()
# Ordenando o Data Frame pelo Número de Mortos, do maior para o menor

df = df.sort_values(by=['Número de Mortos'],ascending=[False])

df.head()
# Verificando a existência de valores nulos no Data Frame

df.isnull().sum()
# Verificando os 10 ataques com o maior número de mortos

df.nlargest(10,'Número de Mortos')
# Verificando os 10 ataques com o menor número de mortos

df.nsmallest(10,'Número de Mortos')
# Verificando os 10 ataques com o maior número de feridos

df.nlargest(10,'Número de Feridos')
# Verificando os 10 ataques com o menor número de feridos

df.nsmallest(10,'Número de Feridos')
# Acrescentando a coluna número de vítimas, cujo resultado é a soma do número de mortos 

# mais o número de feridos

df['Número de Vítimas']=df['Número de Mortos']+df['Número de Feridos']
# Verificando os 10 ataques com o maior número de vítimas

df.nlargest(10,'Número de Vítimas')
# Verificando os 10 ataques com o menor número de vítimas

df.nsmallest(10,'Número de Vítimas')
# Analisando a quantidade de ataques por ano

df["Ano"].value_counts()
# Analisando a Estatística Descritiva dos números de mortos, feridos e vítimas

df[['Número de Mortos','Número de Feridos','Número de Vítimas']].describe()
# Importando as bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns

import folium

%matplotlib inline
# Verificando possíveis correlações entre as varáveis númericas

f,ax=plt.subplots(figsize=(15,6))

sns.heatmap(df.corr(),annot=True,fmt='.2f',ax=ax,linecolor='black',lw=.7)
# Plotando gráfico de barras com o número de ataques registrados por ano

plt.subplots(figsize=(15,6))

sns.countplot('Ano',data=df)

plt.xticks(rotation=90)

plt.title('Número de Ataques Registrados por Ano')

plt.show()
# Plotando gráfico de barras com o número de ataques registrados por mês

plt.subplots(figsize=(15,6))

sns.countplot('Mês',data=df)

plt.xticks(rotation=90)

plt.title('Número de Ataques Registrados por Mês')

plt.show()
# Plotando gráfico de barras com o número de ataques registrados por tipo de arma

plt.subplots(figsize=(15,6))

sns.countplot('Tipo de Arma',data=df)

plt.xticks(rotation=90)

plt.title('Número de Ataques Registrados por Tipo de Arma')

plt.show()
# Plotando gráfico de barras com o número de ataques registrados por tipo de alvo

plt.subplots(figsize=(15,6))

sns.countplot('Tipo de Alvo',data=df)

plt.xticks(rotation=90)

plt.title('Número de Ataques Registrados por Tipo de Alvo')

plt.show()
# Plotando gráfico de linhas com o número de mortos registrados por ano e região

plt.subplots(figsize=(15,6))

sns.lineplot(x='Ano', y='Número de Mortos', hue='Região', data=df)

plt.xticks(rotation=90)

plt.title('Número de Mortos Registrados por Ano e Região')

plt.show()
# Plotando gráfico de barras com o número de vítimas, feridos e mortos por região

sns.set(style='whitegrid')

f,ax=plt.subplots(figsize=(8,15))

sns.set_color_codes('pastel')

sns.barplot(x='Número de Vítimas',y='Região',data=df,label='Número de Vítimas',color='b')

sns.set_color_codes("muted")

sns.barplot(x='Número de Feridos',y='Região',data=df,label='Número de Feridos',color='b')

sns.set_color_codes("dark")

sns.barplot(x='Número de Mortos',y='Região',data=df,label='Número de Mortos',color='b')

ax.legend(ncol=2, loc='lower right', frameon=True)

ax.set(xlim=(0,24),ylabel="",xlabel="Número de Vítimas, Feridos e Mortos Registrados por Região")

sns.despine(left=True, bottom=True)
# Plotando os 1000 ataques com o maior número de mortos no globo terrestre

lt = df[df['Latitude'].notnull() & df['Longitude'].notnull()]['Latitude'][:1000].values

lg = df[df['Latitude'].notnull() & df['Longitude'].notnull()]['Longitude'][:1000].values

mapa = folium.Map(location=[30,0],zoom_start=2)

for i,j in zip(lt,lg):

    folium.Marker([i,j]).add_to(mapa)

mapa