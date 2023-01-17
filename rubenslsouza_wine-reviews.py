# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#criando os dataframes

wine1 = pd.read_csv ('/kaggle/input/winemag-data_first150k.csv')

wine2 = pd.read_csv ('/kaggle/input/winemag-data-130k-v2.csv')

wine3 = pd.read_json('/kaggle/input/winemag-data-130k-v2.json')
wine1.head().T
wine2.head().T
wine3.head().T
wine1.info()
wine2.info()
wine3.info()
#avaliando a similariedade entre wine2 e wine3

df = pd.concat([wine2,wine3])

df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

df.info()
#removendo as duplicidades

df.drop_duplicates(inplace=True)

df.info()
#analisando o conteúdo de wine1

#removendo coluna 'unnamed'

wine1.drop(wine1.columns[wine1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

wine1.sample(20)
#analisando o conteúdo de wine2

wine2 = df

wine2.sample(20)
# obtendo linhas com paises nulos data wine1

wine1[wine1['country'].isnull()]

wine2[wine2['country'].isnull()]
#deletando as colunas com conunry nulo

wine1.dropna(subset=['country'] , inplace=True)

wine2.dropna(subset=['country'] , inplace=True)

wine1.info()

wine2.info()
# obtendo linhas com designação nula data wine1

wine1[wine1['designation'].isnull()]
# obtendo linhas com região 1 nula e região 2 não nula

wine3 = wine1[wine1['region_1'].isnull()]

wine3[wine3['region_2'].notnull()]
#removendo região 2

#removendo dados da wine2 que não existem na wine1

#wine1.drop(['designation','region_2'], axis=1, inplace=True)

#wine2.drop(['designation','region_2','taster_name', 'taster_twitter_handle', 'title'], axis=1, inplace=True)

wine1.drop(['region_2'], axis=1, inplace=True)

wine2.drop(['region_2','taster_name', 'taster_twitter_handle'], axis=1, inplace=True)

wine1.info()

wine2.info()
# obtendo linhas com região 1 nula e região 2 não nula

wine2[wine2['variety'].isnull()]
wine2.dropna(subset=['variety'] , inplace=True)

wine2.info()
#analisando a região 1 do dataframe wine1

wine1[wine1['province'] == 'Maipo Valley']
wine1 = wine1.assign(region_1=np.where(wine1.region_1.isnull(), wine1.region_1.ffill(), wine1.province))

wine1.sample(20)
wine2 = wine2.assign(region_1=np.where(wine2.region_1.isnull(), wine2.region_1.ffill(), wine2.province))

wine2.sample(20)
wine1.info()

wine2.info()
wine3 = pd.concat([wine1,wine2])

wine3.info()
#verificando valores duplicados

# wine3.duplicated()

wine3[wine3.duplicated(subset=['country', 'description', 'designation', 'points', 'price', 

                               'province', 'region_1', 'variety','winery',])]
#removendo as duplicidades

wine3.drop_duplicates(subset=['country', 'description', 'designation', 'points', 'price', 

                               'province', 'region_1', 'variety','winery'], inplace=True)

wine3.info()
wine3[wine3['designation'] == 'Altenberg de Bergheim Grand Cru']
wine3.describe()
#quantidade de vinhos por país

wine_country = wine3.groupby('country').count()[['winery']].reset_index()

wine_country
#quantidade de vinhos por província

df['province'].value_counts()
#quantidade de vinhos por vinícola

df['winery'].value_counts()
# os 10 vinhos mais caros

wine3.nlargest(10,'price').T
# os melhores vinhos

wine3[wine3['points'] == 100]
#médias por países

country_mean = wine3.groupby('country').agg({'points': ['min', 'max', 'mean']}).reset_index()

country_mean
#formatando o join por países

countries = pd.merge(wine_country, country_mean, on='country')

countries.rename(columns={ countries.columns[1]: "count", countries.columns[2]: "point_min", countries.columns[3]: "point_max", countries.columns[4]: "point_mean" }, inplace = True)

countries
#adicionando variável diferença entre a maior e menor pontuação por paises

countries = countries.assign(dif_pont = countries['point_max'] - countries['point_min'])

countries
#importando as bibliotecas gráficas

import seaborn as sns

import matplotlib.pyplot as plt
#gráfico por paises

plt.figure(figsize=(15,5))

sns.boxplot(wine3['country'], wine3['points'])

plt.xticks(rotation=90)

plt.locator_params(axis='y', nbins=20)

plt.show()
# mesma análise por região

wine_region = wine3.groupby('region_1').count()[['country']].reset_index()

region_mean = wine3.groupby('region_1').agg({'points': ['min', 'max', 'mean']}).reset_index()

region = pd.merge(wine_region, region_mean, on='region_1')

region.rename(columns={ region.columns[1]: "count", region.columns[2]: "point_min", region.columns[3]: "point_max", region.columns[4]: "point_mean" }, inplace = True)

region = region.assign(dif_pont = region['point_max'] - region['point_min'])

region
# 20 maiores regiões produtoras 

region_top20 = region.nlargest(20,'count')

region_top20

regions_by_top = pd.merge(wine3, region_top20, on='region_1')

regions_by_top

plt.figure(figsize=(15,5))

sns.boxplot(regions_by_top['region_1'], wine3['points'])

plt.xticks(rotation=90)

plt.locator_params(axis='y', nbins=20)

plt.show()
# mesma análise por vinícola

wine_winery = wine3.groupby('winery').count()[['country']].reset_index()

winery_mean = wine3.groupby('winery').agg({'points': ['min', 'max', 'mean']}).reset_index()

winery = pd.merge(wine_winery, winery_mean, on='winery')

winery.rename(columns={ winery.columns[1]: "count", winery.columns[2]: "point_min", winery.columns[3]: "point_max", winery.columns[4]: "point_mean" }, inplace = True)

winery = winery.assign(dif_pont = winery['point_max'] - winery['point_min'])

winery
# 20 maiores vinícolas

winery_top20 = winery.nlargest(20,'count')

winery_top20

winery_by_top = pd.merge(wine3, winery_top20, on='winery')

winery_by_top

plt.figure(figsize=(15,5))

sns.boxplot(winery_by_top['winery'], wine3['points'])

plt.xticks(rotation=90)

plt.locator_params(axis='y', nbins=20)

plt.show()