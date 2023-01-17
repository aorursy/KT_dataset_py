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
#Carregando os dados

dados_2015= pd.read_csv('/kaggle/input/world-happiness/2015.csv')

dados_2016= pd.read_csv('/kaggle/input/world-happiness/2016.csv')

dados_2017= pd.read_csv('/kaggle/input/world-happiness/2017.csv')
#Tamanho dos dataframes

dados_2015.head()
# pegando o pior rank

dados_2015.nsmallest(5, 'Happiness Score')
#Filtrando o brazil

dados_2015[dados_2015['Country']=='Brazil']['Happiness Rank']
#Filtrando o brazil

dados_2016[dados_2016['Country']=='Brazil']['Happiness Rank']
#Filtrando o brazil

dados_2017[dados_2017['Country']=='Brazil']['Happiness.Rank'],['Country']
frames = [dados_2015, dados_2016, dados_2017]

result = pd.merge(dados_2015, dados_2016, on=['Country'])



#result = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,sort=False,

#          levels=None, names=None, verify_integrity=False, copy=True)



result.head().T
result[result['Country']=='Brazil'].T
import matplotlib.pyplot as plt

import seaborn as sns



dados_2015.plot(title='2015', kind='scatter', x='Economy (GDP per Capita)', y='Happiness Score', color='red')
sns.stripplot(x='Region', y='Happiness Score', data=dados_2015)

plt.xticks(rotation=90)
# correlação dos atributos

dados_2016.corr()
#grafico de correlação dos atributos

f, ax = plt.subplots(figsize=(16,6))

sns.heatmap(dados_2016.corr(), annot=True, fmt='.2f', linecolor='black', lw=.7, ax=ax)
dados_2016.describe()
#criando uma nova coluna happy_quality

# definindo os valores

q3 = dados_2016['Happiness Score'].quantile(0.75)

q2 = dados_2016['Happiness Score'].quantile(0.5)

q1 = dados_2016['Happiness Score'].quantile(0.25)



happy_quality =[]



# classificando os valores

for valor in dados_2016['Happiness Score']:

    if valor >= q3:

        happy_quality.append('Muito Alto')

    elif valor < q3 and valor >= q2:

        happy_quality.append('Alot')

    elif valor <q2 and valor >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')



#incluindo nova coluna happy_quality

dados_2016['happy_quality']= happy_quality

dados_2016
# boxplot usando happy_quality



plt.figure(figsize=(7,7))

sns.boxplot(dados_2016['happy_quality'], dados_2016['Economy (GDP per Capita)'])
#pegando os cinco menores

dados_2016.nsmallest(5, 'Economy (GDP per Capita)')
# swarmplot usando happy_quality



plt.figure(figsize=(7,7))

sns.swarmplot(dados_2016['happy_quality'], dados_2016['Economy (GDP per Capita)'])
# correlação entre  Health e PIB

plt.figure(figsize=(10,7))



sns.scatterplot(dados_2016['Health (Life Expectancy)'], dados_2016['Economy (GDP per Capita)'], 

                hue=dados_2016['happy_quality'], 

                style=dados_2016['happy_quality'])
