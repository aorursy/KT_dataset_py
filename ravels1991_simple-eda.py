# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import unicodedata

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cusersmarildownloadsblackcsv/black.csv', delimiter=';', encoding = "latin1", low_memory=False)

df.head()
print(f'Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}')
list(df.isnull().mean())
df = df.iloc[:, :12]

df.head(20)
df.info()
df.isnull().mean()*100
#drop all na values

df.dropna(how='any', axis=0, inplace=True)

#print new shape of the data

print(f'Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}')
#change columns types

df[['nome', 'uf', 'censo', 'geom']] = df[['nome', 'uf', 'censo', 'geom']].astype(str)
#drop cordinates rows

a = df[df['censo'].map(len) > 5]

df = df.drop(a.index)

df.reset_index(drop=True, inplace=True)

df.head()
#print thew new shape

print(f'Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}')
#change uf and name to ascii character

df['uf'].value_counts()
#this lambda expression has been copy from this stackoverflow page

#https://stackoverflow.com/questions/49891778/conversion-utf-to-ascii-in-python-with-pandas-dataframe

df['nome'] = df['nome'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

df['uf'] = df['uf'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

df['uf'].value_counts()
df['uf'].nunique()
#Brazil don't have 34 uf, we have 26(and DISTRITO FEDERAL)

df['uf'] = df['uf'].str.upper()

df['nome'] = df['nome'].str.upper()

df['uf'].nunique()
#almost there

print(df['uf'].unique())

df['uf'].replace({'PIAUA': 'PIAUI', 'PARAABA':'PARAIBA'}, inplace=True)

print(df['uf'].nunique())
#create a new colum with region of the State.

SUL = ['SANTA CATARINA', 'RIO GRANDE DO SUL', 'PARANA']

SUDESTE = ['MINAS GERAIS', 'RIO DE JANEIRO', 'SAO PAULO', 'ESPIRITO SANTO']

CENTRO_OESTE = ['DISTRITO FEDERAL', 'GOIAS', 'MATO GROSSO', 'MATO GROSSO DO SUL' ]

NORDESTE = ['ALAGOAS', 'BAHIA', 'CEARA', 'MARANHAO', 'PARAIBA', 'PERNAMBUCO', 'PIAUI', 'RIO GRANDE DO NORTE', 'SERGIPE']

NORTE = ['ACRE', 'AMAPA', 'AMAZONAS', 'PARA', 'RONDANIA', 'RORAIMA', 'TOCANTINS']
def regiao(row):

    if row in SUL:

        return 'SUL'

    if row in SUDESTE:

        return 'SUDESTE'

    if row in CENTRO_OESTE:

        return 'CENTRO_OESTE'

    if row in NORDESTE:

        return 'NORDESTE'

    if row in NORTE:

        return 'NORTE'
df['regiao'] = df['uf'].apply(regiao)

df.head()
#change the type of columns and fixing per_pessoas_pretas column

df[['pop_tot', 'pessoas_pretas']] = df[['pop_tot', 'pessoas_pretas']].astype(float)

df['per_pessoas_pretas'] = round(df['pessoas_pretas'] / df['pop_tot']*100, 2)

df.head()
df['geom'].head()
#using regex to remove string character and replace parentheses 

df['geom'] = df['geom'].str.replace(r"[a-zA-Z]", '').str.replace("(", '')

df.head()
#making lon and lat columns

df[['del1', 'lon', 'lat', 'del2']] = df['geom'].str.split(' ', 3, expand=True)

df.head()
#chacing type to float

df['lat'] = df['lat'].str.replace(',', '')

df[['lat', 'lon']] = df[['lat', 'lon']].astype(float)
#drop columns

df = df.drop(['fid', 'gid', 'descricao', 'legenda', 'classe', 'geom', 'del1', 'del2'], axis=1)

df.head()
ax = df.groupby('uf')['per_pessoas_pretas'].max().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Maximo % de pessoas negras por Estado')

plt.xlabel('% pessoas pretas')

plt.ylabel('Estado')

plt.show()
ax = df.groupby('uf')['per_pessoas_pretas'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Media % pessoas negras por Estado')

plt.xlabel('% pessoas pretas')

plt.ylabel('Estado')

plt.show()
ax = df.groupby('uf')['per_pessoas_pretas'].min().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), 

                                                                                  title='Minimo % pessoas negras por Estado')

plt.xlabel('% pessoas pretas')

plt.ylabel('Estado')

plt.show()
ax = df.groupby('regiao')['per_pessoas_pretas'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Media % pessoas negras por Regiao')

plt.xlabel('% pessoas pretas')

plt.ylabel('Regiao')

plt.show()
ax = df.groupby('regiao')['per_pessoas_pretas'].min().plot(kind='barh', figsize=(12,8),

                                                          title='Minima % de pessoas negras por regiao')

plt.xlabel('% pessoas pretas')

plt.ylabel('Regiao')

plt.show()
ax = df.groupby('regiao')['pop_tot', 'pessoas_pretas'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Populacao Total x Pessoas Pretas')

plt.xlabel('Regiao')

plt.ylabel('Log Scale Populacao')

plt.show()
ax = df.groupby('uf')['pop_tot', 'pessoas_pretas'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Populacao Total x Pessoas Pretas', logx=True, linewidth=3)

plt.xlabel('Log Scale Populacao')

plt.ylabel('Regiao')

plt.show()
ax = df.groupby('uf')['per_pessoas_pretas'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(20,6), 

                                                                                    title='Media % pessoas negras por estado')

plt.xlabel('% pessoas pretas')

plt.ylabel('Estado')

plt.show()
g = sns.lmplot(x="lon", y="lat", data=df,

           fit_reg=False, scatter_kws={"s": 30}, hue='regiao', height=10)

plt.title('MAPA BRASIL')

plt.show()
plt.figure(figsize=(20,12))

g = sns.scatterplot(x='lon', y='lat', data=df, hue='uf')

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
df.groupby('regiao')['per_pessoas_pretas'].min()
df.groupby('regiao')['per_pessoas_pretas'].max()
df[(df['regiao'] == 'NORDESTE') & (df['per_pessoas_pretas'] == 50.65)]
df[(df['regiao'] == 'SUL') & (df['per_pessoas_pretas'] == 0)]
plt.figure(figsize=(20,12))

g = sns.scatterplot(x='lon', y='lat', data=df, hue='uf')

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);

#this annotate has been copy from this stackoverflow page

#https://stackoverflow.com/questions/39147492/annotate-seaborn-factorplot

plt.annotate('Antônio Cardoso', xy=(-39.134271, -12.283881), xytext=(-39.134271, -12.283881),

             arrowprops=dict(facecolor='red', shrink=0.05, headwidth=8))

plt.annotate('Cunhataí', xy=(-53.097716, -26.925436), xytext=(-53.097716, -26.925436),

             arrowprops=dict(facecolor='red', shrink=0.05, headwidth=8))

plt.show()