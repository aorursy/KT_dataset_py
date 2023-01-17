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

#Carregando dos Dados

dados_2015 = pd.read_csv('../input/2015.csv')

dados_2016 = pd.read_csv('../input/2016.csv')

dados_2017 = pd.read_csv('../input/2017.csv')
#Olhando o tamanho dos dataframes

print('2015: ',dados_2015.shape)

print('2015: ',dados_2016.shape)

print('2015: ',dados_2017.shape)
#Olhando os dados

dados_2015.head()
#Olhando os dados

dados_2016.head()
#Olhando os dados

dados_2017.head()
# Verificando os nomes das colunas

dados_2015.columns
# Verificando os nomes das colunas

dados_2016.columns
# Verificando os nomes das colunas

dados_2017.columns
# Vamos manter apenas as colunas mais importantes e Apagar o resto

dados_2015.drop(columns='Standard Error' , inplace=True)

dados_2015.columns
# Vamos manter apenas as colunas mais importantes e Apagar o resto
dados_2016.drop(columns=['Lower Confidence Interval','Upper Confidence Interval'] , inplace=True)

dados_2016.columns
dados_2017.drop(columns=['Whisker.high','Whisker.low'] , inplace=True)

dados_2017.columns



# Novos nomes de colunas

nomes_2015_2016 = ['country','region','happiness_rank','happiness_score','economy','family','health','fredom','corruption',

             'generosity','dystopia']



nomes_2017 = ['country','happiness_rank','happiness_score','economy', 'family','health','fredom','generosity','corruption',

             'dystopia']



#Alterando os nomes das colunas

dados_2015.columns = nomes_2015_2016

dados_2016.columns = nomes_2015_2016

dados_2017.columns = nomes_2017

#Os 5 piores paises por ano

dados_2015.nsmallest(5, 'happiness_score')
dados_2016.nlargest(5,'happiness_rank')
dados_2017.nlargest(5,'happiness_rank')
#Qual a posição do BRasil a cada ano?

print('===== Brasil ======')

print('2015: ', dados_2015[dados_2015['country'] == 'Brazil']['happiness_rank'])

print('2016: ', dados_2016[dados_2016['country'] == 'Brazil']['happiness_rank'])

print('2017: ', dados_2017[dados_2017['country'] == 'Brazil']['happiness_rank'])
# importando as bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Vamos relacionar happiness_score e PIB

dados_2015.plot(title='2015',kind='scatter', x='economy', y='happiness_score',

               alpha=0.5, color='red')



dados_2016.plot(title='2016',kind='scatter', x='economy', y='happiness_score',

               alpha=0.5, color='green')



dados_2017.plot(title='2017',kind='scatter', x='economy', y='happiness_score',

               alpha=0.5, color='blue')

plt.show
# Vamos relacionar happiness_score e região

sns.stripplot(x='region',y='happiness_score', data=dados_2015, jitter=True)

plt.xticks(rotation=90)
# correlação

dados_2015.corr()
# Vamos plotar as correlações

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(dados_2015.corr(),annot=True, fmt='.2f', ax=ax, linecolor='black',lw=.7)
# Qual o fator mais afeta o happiness_score por região 2015

#dados_2015['region'].value_counts()

dados_2015['region'].unique()

region = list(dados_2015['region'].unique())

economy = []

family = []

health = []

fredom = []

corruption = []



for r in region:

    df = dados_2015[dados_2015['region'] == r]

    economy.append(df['economy'].mean())

    family.append(df['family'].mean())

    health.append(df['health'].mean())

    fredom.append(df['fredom'].mean())

    corruption.append(df['corruption'].mean())

    

economy

    

    

    

    

    
# Plotar os valores



plt.figure(figsize=(10,5))

sns.barplot(x=economy, y=region, color='pink', label='economy')

sns.barplot(x=family, y=region, color='red', label='family')

sns.barplot(x=health, y=region, color='blue', label='health')

sns.barplot(x=fredom, y=region, color='orange', label='fredom')

sns.barplot(x=corruption, y=region, color='black', label='corruption')

plt.legend()

region
# Criando uma nova coluna

#happy_quality

q3 = dados_2017['happines_score'].quantile(0.75)

q2 = dados_2017['happines_score'].quantile(0.5)

q1 = dados_2017['happines_score'].quantile(0.25)



happy_quality = []



for i in dados_2017['happiness_score']:

    if i >= q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i < q2 and i >= q1:

        happy_quality.append('Normal')

    else: 

        happy_quality.append('Baixo')    



dados_2017['happy_quality'] = happy_quality

    
#Criando uma coluna nova 

#happy_quality



q3 = dados_2017['happiness_score'].quantile(0.75)

q2 = dados_2017['happiness_score'].quantile(0.5)

q1 = dados_2017['happiness_score'].quantile(0.25)



happy_quality = []



for i in dados_2017['happiness_score']:

    if i > q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i < q2 and i >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')



#Criando a coluna no dataframe e atribuindo os valores do df happy_quality

dados_2017['happy_quality'] = happy_quality



plt.figure(figsize=(7,7))

sns.boxplot(dados_2017['happy_quality'], dados_2017['economy'])
plt.figure(figsize=(7,7))

sns.scatterplot(dados_2017['fredom'], dados_2017['economy'],

                hue=dados_2017['happy_quality'],

                style=dados_2017['happy_quality'])