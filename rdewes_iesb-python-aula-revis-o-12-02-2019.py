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
df2015 = pd.read_csv('../input/2015.csv')

df2016 = pd.read_csv('../input/2016.csv')

df2017 = pd.read_csv('../input/2017.csv')
print('2015: ', df2015.shape)

print('2016: ', df2016.shape)

print('2017: ', df2017.shape)
df2015.head()
df2016.head()
df2017.head()
df2015.columns
df2016.columns
df2017.columns
# Vamos manter apenas as colunas mais importantes

df2015 = df2015.drop(columns='Standard Error')

df2015.columns
df2016 = df2016.drop(columns=['Lower Confidence Interval', 'Upper Confidence Interval'])
df2017 = df2017.drop(columns=['Whisker.high', 'Whisker.low'])
# Padronizando os nomes

nomes_2015 = ['country', 'region', 'happiness_rank', 'happiness_score',

             'economy', 'family', 'health', 'freedom', 'corruption',

             'generosity', 'distopy']



nomes_2016 = ['country', 'region', 'happiness_rank', 'happiness_score',

             'economy', 'family', 'health', 'freedom', 'corruption',

             'generosity', 'distopy']



nomes_2017 = ['country', 'happiness_rank', 'happiness_score',

             'economy', 'family', 'health', 'freedom', 'generosity',

             'corruption', 'distopy']



# Renomeando as colunas

df2015.columns = nomes_2015

df2016.columns = nomes_2016

df2017.columns = nomes_2017
df2015.nsmallest(5, 'happiness_score')
df2015['year'] = 2015

df2016['year'] = 2016

df2017['year'] = 2017

todos = pd.concat([df2015, df2016, df2017], sort=True)
dd
# Verificando a posição do Brasil

print(todos[todos.country == 'Brazil'][['happiness_rank','year']])
import matplotlib.pyplot as plt

import seaborn as sns
#plt.subplot(sharex=True, sharey=True)

df2015.plot(title='2015', kind='scatter', x='economy', y='happiness_score', 

            alpha=0.5, color='red')

plt.xlabel('PIB')

plt.ylabel('Felicidade')

df2016.plot(title='2016', kind='scatter', x='economy', y='happiness_score', 

            alpha=0.5, color='green')

plt.xlabel('PIB')

plt.ylabel('Felicidade')

df2017.plot(title='2017', kind='scatter', x='economy', y='happiness_score', 

            alpha=0.5, color='blue')

plt.xlabel('PIB')

plt.ylabel('Felicidade')

plt.show()
sns.stripplot(y='region', x='happiness_score', data=df2015, jitter=True)

#plt.xticks(rotation=45)
df2015.corr()
f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(todos.corr(), annot=True, fmt='.2f', ax=ax)
df2015['region'].unique()
%%time

dados = df2015.groupby('region')[['economy', 'family', 'health', 'freedom', 'corruption']].mean()

dados = dados.reset_index()

dados
plt.figure(figsize=(10,5))

sns.barplot(x='economy', y='region', data=dados, color='pink', label='economy')

sns.barplot(x='family', y='region', data=dados, color='red', label='family')

sns.barplot(x='health', y='region', data=dados, color='blue', label='health')

sns.barplot(x='freedom', y='region', data=dados, color='orange', label='freedom')

sns.barplot(x='corruption', y='region', data=dados, color='black', label='corruption')

plt.legend()
%%time

# Criando uma coluna nova



q3 = df2017['happiness_score'].quantile(0.75)

q2 = df2017['happiness_score'].quantile(0.5)

q1 = df2017['happiness_score'].quantile(0.25)



happy_quality = []



for i in df2017['happiness_score']:

    if i >= q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i < q2 and i >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')



df2017['happy_quality'] = happy_quality
%%time

q3 = df2017['happiness_score'].quantile(0.75)

q2 = df2017['happiness_score'].quantile(0.5)

q1 = df2017['happiness_score'].quantile(0.25)



def classifica(i):

    if i >= q3:

        return('Muito Alto')

    elif i < q3 and i >= q2:

        return('Alto')

    elif i < q2 and i >= q1:

        return('Normal')

    else:

        return('Baixo')



df2017['happy_quality'] = df2017['happiness_score'].map(classifica)
plt.figure(figsize=(7,7))

sns.boxplot(df2017['happy_quality'], df2017['economy'])
sns.swarmplot(df2017['happy_quality'], df2017['economy'])
sns.scatterplot(df2017['freedom'], df2017['economy'], hue=df2017['happy_quality'],

               style=df2017['happy_quality'])