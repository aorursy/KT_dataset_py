import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

os.chdir('/kaggle/input/gas-prices-in-brazil/')
df = pd.read_csv('2004-2019.tsv',sep='\t')
df.head()
df['PRODUTO'].unique()
df.drop('Unnamed: 0',axis=1,inplace=True)
b = df['PRODUTO'].unique()
b
k = []

for i in range(len(b)):

    k.append(df[df['PRODUTO'] == b[i]].groupby('ANO')['PREÇO MÉDIO REVENDA'].mean())
a = df[df['PRODUTO'] == b[1]].groupby('ANO')['PREÇO MÉDIO REVENDA'].mean()
x = a.index

plt.figure(figsize=(9,5))

plt.grid(True)

plt.xlabel('Ano')

plt.ylabel('Preco do combustivel (em reais)')

plt.title('Evolucao do preco medio anual de combustiveis no Brasil')

plt.plot(x,k[0].values,x,k[1].values,x,k[3].values,x,k[4].values)

plt.legend([b[0],b[1],b[3],b[4]])

plt.show()
plt.figure(figsize=(9,5))

plt.grid(True)

plt.xlabel('Ano')

plt.ylabel('Preco do combustivel (em reais)')

plt.title('Evolucao do preco medio anual do GLP no Brasil')

plt.plot(x,k[2].values)

#plt.legend([b[0],b[1],b[3],b[4]])

plt.show()
df.columns
Z = pd.DataFrame(df[df['PRODUTO'] == b[1]].groupby(['REGIÃO','ANO'])['PREÇO MÉDIO REVENDA'].mean())
c = df['REGIÃO'].unique()

c
hh = []

for i in range(len(c)):

    hh.append(df[(df['PRODUTO'] == b[1]) & (df['REGIÃO'] == c[i])].groupby(['ANO'])['PREÇO MÉDIO REVENDA'].mean())
plt.figure(figsize=(9,5))

plt.grid(True)

plt.xlabel('Ano')

plt.ylabel('Preco do combustivel (em reais)')

plt.title('Evolucao do preco medio anual de '+ b[1] +' por regiao')

plt.plot(x,hh[0].values,x,hh[1].values,x,hh[2].values,x,hh[3].values)

plt.legend([c[0],c[1],c[2],c[3]])

plt.show()
df['PRECO MEDIO REVENDA'] = df['PREÇO MÉDIO REVENDA']

dff = df.drop('PREÇO MÉDIO REVENDA',axis = 1)
for i in range(len(b)):

    sns.set()

    plt.figure(figsize=(14,8))

    plt.title('Perfil de preco medio de revenda de '+b[i]+' no Brasil')

    j = sns.boxplot(x='ANO',y = 'PRECO MEDIO REVENDA', data = dff[dff['PRODUTO'] == b[i]])

    j.set_xticklabels(j.get_xticklabels(), rotation=45)

    plt.show()
f = df[df['ANO'] == 2019]

f.head()
f.info()
f.describe()
n = f[f['PRODUTO'] == b[1]]

n[n['PREÇO MÉDIO REVENDA'] == n['PREÇO MÉDIO REVENDA'].max()] #dataset da gasolina mais cara em 2019
f.groupby(['REGIÃO','PRODUTO'])['PREÇO MÉDIO REVENDA'].mean() #preço médio dos combustíveis por região
f.groupby(['REGIÃO','PRODUTO'])['PREÇO MÁXIMO REVENDA'].max() #preço máximo da gasolina por região
f.groupby(['REGIÃO','PRODUTO'])['PREÇO MÍNIMO REVENDA'].min() #preço mínimo de combustíveis por região
n['REGIAO'] = n['REGIÃO']

n['PRECO MEDIO REVENDA'] = n['PREÇO MÉDIO REVENDA']
plt.figure(figsize=(10,6))

plt.title('Distribuicao de preco da gasolina em 2019 por regiao')

sns.barplot(x = 'REGIAO',y = 'PRECO MEDIO REVENDA',data=n)

plt.show()
for i in range(len(b)):

    plt.figure(figsize=(17,6))

    plt.title('Distribuicao de preco de '+b[i]+' em 2019 por estado')

    jk = sns.boxplot(x='ESTADO',y = 'PRECO MEDIO REVENDA', data = f[f['PRODUTO'] == b[i]])

    jk.set_xticklabels(jk.get_xticklabels(), rotation=45)

    plt.show()
ff = f[f['MÊS'] == 6]

ff['PRECO MAXIMO REVENDA'] = ff['PREÇO MÁXIMO REVENDA']
ff.groupby(['REGIÃO','PRODUTO'])['PREÇO MÉDIO REVENDA'].mean() #preço médio dos combustíveis por região
ff.groupby(['REGIÃO','PRODUTO'])['PREÇO MÁXIMO REVENDA'].max() #preço máximo da gasolina por região
ff.groupby(['REGIÃO','PRODUTO'])['PREÇO MÍNIMO REVENDA'].min() #preço mínimo de combustíveis por região
fff = ff[ff['PRODUTO'] == b[0]]

fff[fff['PREÇO MÁXIMO REVENDA'] == fff['PREÇO MÁXIMO REVENDA'].max()]
fff = ff[ff['PRODUTO'] == b[0]]

fff[fff['PREÇO MÍNIMO REVENDA'] == fff['PREÇO MÍNIMO REVENDA'].min()]
fff = ff[ff['PRODUTO'] == b[1]]

fff[fff['PREÇO MÁXIMO REVENDA'] == fff['PREÇO MÁXIMO REVENDA'].max()]
fff = ff[ff['PRODUTO'] == b[1]]

fff[fff['PREÇO MÍNIMO REVENDA'] == fff['PREÇO MÍNIMO REVENDA'].min()]
fff = ff[ff['PRODUTO'] == b[4]]

fff[fff['PREÇO MÁXIMO REVENDA'] == fff['PREÇO MÁXIMO REVENDA'].max()]
fff = ff[ff['PRODUTO'] == b[4]]

fff[fff['PREÇO MÍNIMO REVENDA'] == fff['PREÇO MÍNIMO REVENDA'].min()]
for i in range(len(b)):

    plt.figure(figsize=(17,6))

    plt.title('Distribuicao de preco de '+b[i]+' em junho por estado')

    jk = sns.boxplot(x='ESTADO',y = 'PRECO MEDIO REVENDA', data = ff[ff['PRODUTO'] == b[i]])

    jk.set_xticklabels(jk.get_xticklabels(), rotation=45)

    plt.show()
for i in range(len(b)):

    plt.figure(figsize=(17,6))

    plt.title('Distribuicao de preco maximo de revenda de '+b[i]+' em 2019 por estado')

    jk = sns.barplot(x = 'ESTADO',y = 'PRECO MAXIMO REVENDA',data=ff[ff['PRODUTO'] == b[i]])

    jk.set_xticklabels(jk.get_xticklabels(), rotation=45)

    plt.show()