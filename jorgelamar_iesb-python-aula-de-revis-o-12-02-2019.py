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

dados_2015 = pd.read_csv('../input/2015.csv')

dados_2016 = pd.read_csv('../input/2016.csv')

dados_2017 = pd.read_csv('../input/2017.csv')
# Olhando o tamanho dos dataframes

print('2015: ',dados_2015.shape)

print('2016: ',dados_2016.shape)

print('2017: ',dados_2017.shape)
#Olhandos os dados

dados_2015.head()
dados_2016.head()
dados_2017.head()
# Verificar o nome das colunas 

dados_2015.columns
dados_2016.columns
dados_2017.columns
# Vamos manter apenas as colunas mais importantes

dados_2015.drop(columns='Standard Error',inplace=True)

dados_2015.columns
dados_2016.drop(columns=['Lower Confidence Interval', 'Upper Confidence Interval'],inplace=True)

dados_2016.columns
dados_2017.drop(columns=['Whisker.high','Whisker.low'],inplace=True)

dados_2017.columns
# Novos nomes de colunas

nomes_2015_2016 = ['country','region','happiness_renk','happiness_scores',

           'economy','family','health','freedom','corruption',

                 'generosity','dystopia']

nomes_2017 = ['country','happiness_renk','happiness_scores',

           'economy','family','health','freedom','generosity','corruption',

            'dystopia']



#Alterando os nomes das colunas



dados_2015.columns = nomes_2015_2016

dados_2016.columns = nomes_2015_2016

dados_2017.columns = nomes_2017
#Os 5 piores paises por ano

dados_2015.nsmallest(5, 'happiness_scores')
dados_2016.nlargest(5, 'happiness_renk')
dados_2017.nlargest(5, 'happiness_renk')
# qual a Posição do Brasil a cada ano?

print('=========Brasil========')

print('2015: ', dados_2015[dados_2015['country']=='Brazil']['happiness_renk'])

print('2016: ', dados_2016[dados_2016['country']=='Brazil']['happiness_renk'])

print('2017: ', dados_2017[dados_2017['country']=='Brazil']['happiness_renk'])
# Importando as habilidades gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Vamos relacionar happiness_score e PIB

dados_2015.plot(title='2015', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='red')

dados_2016.plot(title='2016', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='green')

dados_2017.plot(title='2017', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='blue')

plt.show()
# Vamos relacionar happiness_score e PIB

dados_2015.plot(title='2015', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='red')

dados_2016.plot(title='2016', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='green')

dados_2017.plot(title='2017', kind='scatter', x='economy', y='happiness_scores',

                alpha=0.5, color='blue')

plt.xlabel = 'economy'

plt.ylabel = 'happiness_score'

plt.show()
#VAmosm relacionar happiness_scores e regiao

sns.stripplot(x='region',y='happiness_scores', data=dados_2015, jitter=True)

plt.xticks(rotation=90)
#VAmosm relacionar happiness_scores e regiao

sns.stripplot(x='region',y='happiness_scores', data=dados_2015, jitter=True)

#plt.xticks(rotation=90)
# Correlação 

dados_2015.corr()
# VAmos plotar as correlações

sns.heatmap(dados_2015.corr())
# VAmos plotar as correlações

sns.heatmap(dados_2015.corr(), annot=True)
# VAmos plotar as correlações

sns.heatmap(dados_2015.corr(), annot=True, fmt = '.2f')
# VAmos plotar as correlações

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(dados_2015.corr(), annot=True, fmt = '.2f', ax=ax)
# VAmos plotar as correlações

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(dados_2015.corr(), annot=True, fmt = '.2f', ax=ax, linecolor='black', lw=.7)
# Qual o fator mais afeta o happiness_scores por região / 2015

dados_2015['region'].unique()
# Qual o fator mais afeta o happiness_scores por região / 2015

dados_2015['region'].value_counts()
region = list(dados_2015['region'].unique())

economy = []

family = []

health = []

freedom = []

corruption = []



for r in region:

    df = dados_2015[dados_2015['region'] == r]

    economy.append(df['economy'].mean())

    family.append(df['family'].mean())

    health.append(df['health'].mean())

    freedom.append(df['freedom'].mean())

    corruption.append(df['corruption'].mean())        

     

#plotar os valores

plt.figure(figsize=(10,5))

sns.barplot(x=economy, y=region, color='pink', label='economy')

sns.barplot(x=family, y=region, color='red', label='family')

sns.barplot(x=health, y=region, color='blue', label='health')

sns.barplot(x=freedom, y=region, color='orange', label='freedom')

sns.barplot(x=corruption, y=region, color='black', label='corruption')
region = list(dados_2015['region'].unique())

economy = []

family = []

health = []

freedom = []

corruption = []



for r in region:

    df = dados_2015[dados_2015['region'] == r]

    economy.append(df['economy'].mean())

    family.append(df['family'].mean())

    health.append(df['health'].mean())

    freedom.append(df['freedom'].mean())

    corruption.append(df['corruption'].mean())        

     

#plotar os valores

plt.figure(figsize=(10,5))

sns.barplot(x=economy, y=region, color='pink', label='economy')

sns.barplot(x=family, y=region, color='red', label='family')

sns.barplot(x=health, y=region, color='blue', label='health')

sns.barplot(x=freedom, y=region, color='orange', label='freedom')

sns.barplot(x=corruption, y=region, color='black', label='corruption')



plt.legend()
region
economy
family
region = list(dados_2015['region'].unique())

economy = []

family = []

health = []

freedom = []

corruption = []



for r in region:

    df = dados_2015[dados_2015['region'] == r]

    economy.append(df['economy'].mean())

    family.append(df['family'].mean())

    health.append(df['health'].mean())

    freedom.append(df['freedom'].mean())

    corruption.append(df['corruption'].mean())        

     

#plotar os valores

plt.figure(figsize=(10,5))

sns.barplot(x=economy, y=region, color='pink', label='economy')

sns.barplot(x=family, y=region, color='red', label='family')

#sns.barplot(x=health, y=region, color='blue', label='health')

#sns.barplot(x=freedom, y=region, color='orange', label='freedom')

#sns.barplot(x=corruption, y=region, color='black', label='corruption')



plt.legend()
# criando uma coluna nova

# happy_quality

q3 = dados_2017['happiness_scores'].quantile(0.75)

q2 = dados_2017['happiness_scores'].quantile(0.5)

q1 = dados_2017['happiness_scores'].quantile(0.25)



happy_quality = []



for i in dados_2017['happiness_scores']:

    if i >= q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i < q2 and i >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')

        

happy_quality
# criando uma coluna nova

# happy_quality

q3 = dados_2017['happiness_scores'].quantile(0.75)

q2 = dados_2017['happiness_scores'].quantile(0.5)

q1 = dados_2017['happiness_scores'].quantile(0.25)



happy_quality = []



for i in dados_2017['happiness_scores']:

    if i >= q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i < q2 and i >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')

        

dados_2017['happy_quality'] = happy_quality
plt.figure(figsize=(7,7))

sns.boxplot(dados_2017['happy_quality'], dados_2017['economy'])
plt.figure(figsize=(7,7))

sns.swarmplot(dados_2017['happy_quality'], dados_2017['economy'])
plt.figure(figsize=(7,7))

sns.scatterplot(dados_2017['freedom'], dados_2017['economy'])
plt.figure(figsize=(7,7))

sns.scatterplot(dados_2017['freedom'], dados_2017['economy'],

                hue=dados_2017['happy_quality'],

                style=dados_2017['happy_quality'])
plt.figure(figsize=(7,7))

sns.scatterplot(dados_2017['health'], dados_2017['economy'],

                hue=dados_2017['happy_quality'],

                style=dados_2017['happy_quality'])