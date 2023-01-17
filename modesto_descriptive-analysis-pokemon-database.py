# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/pokemon.csv")
df.head()
df.tail()
df.sample(10) # Pegando 10 registros aleatórios na base
df.info()
df.shape
# Descobrindo qual o valor NaN no campo "Name"

df[df['Name'].isna() == True]
df.dropna(subset=['Name'], inplace=True)

df.info()
df.loc[60:70] # Apresentandos os registros de 59 a 69
df.reset_index(inplace=True)

df.loc[60:70]
df.drop(['index', '#'], axis=1, inplace=True) # Removendo as duas colunas index e #

df.head()
df.plot(title='HP', kind='hist', y='HP',

               alpha=0.5, color='red')



df.plot(title='Attack', kind='hist', y='Attack',

               alpha=0.5, color='green')



df.plot(title='Defense', kind='hist', y='Defense',

               alpha=0.5, color='blue')



df.plot(title='Sp. Atk', kind='hist', y='Sp. Atk',

               alpha=0.5, color='black')



df.plot(title='Sp. Def', kind='hist', y='Sp. Def',

               alpha=0.5, color='yellow')



df.plot(title='Speed', kind='hist', y='Speed',

               alpha=0.5, color='orange')



df.plot(title='Generation', kind='hist', y='Generation',

               alpha=0.5, color='orange')



plt.show()
df.describe()
_,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax)

plt.show()
plt.subplots(figsize=(20, 5))

sns.countplot(x='Type 1', data=df)

plt.show()
plt.subplots(figsize=(20, 5))

sns.countplot(x='Type 2', data=df)

plt.show()
plt.subplots(figsize=(20, 5))

sns.boxplot(x='Type 1', y='HP', data=df)

plt.show()
plt.subplots(figsize=(20, 5))

sns.boxplot(x='Type 1', y='Attack', data=df)

plt.show()
plt.subplots(figsize=(20, 5))

sns.boxplot(x='Type 1', y='Defense', data=df)

plt.show()
sns.factorplot(x='Type 1', y='HP', data=df, col='Legendary', kind='box')
df_normal = df[df['Type 1'] == 'Normal']

df_normal.nlargest(20, 'HP')
df['Legendary'].value_counts() # Quantidade de pokemons lendários
# Imprimindo o percentual de lendários em um gráfico de pizza

plt.pie(df['Legendary'].value_counts(), labels=("Normal", "Lendário"), 

        autopct='%1.1f%%', shadow=True, startangle=90)
df_lendarios = df[df['Legendary']]

df_lendarios
plt.subplots(figsize=(20, 5))

sns.violinplot(x='Type 1', y='HP', data=df)

plt.show()
sns.factorplot(x='Generation', y='HP', data=df, col='Type 1', kind='violin')
df['indice_equilibrio'] = df['Attack'] - df['Defense']

df['indice_poder'] = df['HP'] + df['Attack'] + df['Defense']

df.head()
df.plot(title='indice_equilibrio', kind='hist', y='indice_equilibrio',

               alpha=0.5, color='red')



df.plot(title='indice_poder', kind='hist', y='indice_poder',

               alpha=0.5, color='green')
df.nlargest(20, 'indice_equilibrio') # Os 20 Pokemons mais fortes para atacar
df.nsmallest(20, 'indice_equilibrio') # Os 20 Pokemons mais fortes para defender
df_mais_fortes = df.nlargest(20, 'indice_poder') # Os 20 Pokemons mais fortes



# Gerando o plot dos 20 Pokemons mais fortes

plt.subplots(figsize=(10,5))

plt.grid(True, linestyle='--')

plt.title('Pokemons mais fortes')

plt.plot(df_mais_fortes['Name'], df_mais_fortes['indice_poder'], label='Índice poder', marker='o')

plt.xticks(rotation=90)

plt.xlabel('Nome do Pokemon')

plt.ylabel('Ìndice de Poder')

plt.legend()

plt.show()
df_mais_fracos = df.nsmallest(20, 'indice_poder') # Os 20 Pokemons mais fracos



# Gerando o plot dos 20 Pokemons mais fracos

plt.subplots(figsize=(10,5))

plt.grid(True, linestyle='--')

plt.title('Pokemons mais fracos')

plt.plot(df_mais_fracos['Name'], df_mais_fracos['indice_poder'], label='Índice poder', marker='o')

plt.xticks(rotation=90)

plt.xlabel('Nome do Pokemon')

plt.ylabel('Índice de Poder')

plt.legend()

plt.show()
df.describe() # Avaliando as estatisticas básicas dos dois novos campos
df.plot(kind = "scatter",x="indice_poder",y = "indice_equilibrio")