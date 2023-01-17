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
# Criando os dataframes

wcup = pd.read_csv('/kaggle/input/fifa-world-cup/WorldCups.csv')

matches = pd.read_csv('/kaggle/input/fifa-world-cup/WorldCupMatches.csv')

players = pd.read_csv('/kaggle/input/fifa-world-cup/WorldCupPlayers.csv')
#Verificando os tamanhos dos dataframes



print("copas do Mundo:", wcup.shape)

print('Partidas:', matches.shape)

print('Jogadores:', players.shape)
wcup.info()
wcup.head(3).T
matches.info()
matches.sample(5).T

# Excluindo as linhas onde todos os valores são nulos

matches.dropna(how='all', inplace=True)



matches.info()
# verificando as duas partidas com attendance = 850



matches[matches['Attendance'].isnull()]

matches.head()

matches['MatchID'].value_counts()
matches[matches['MatchID'] ==300186490.0]
# usando o duplicated



matches.duplicated()

# Obtendo todos os jogos duplicados

matches[matches.duplicated(keep=False)]
# Eliminando os jogos repetidos



matches.drop_duplicates(inplace=True)



matches.info()
# Quantas partidas por ano?



matches['Year'].value_counts()
### Informações e análises descritivas

matches.describe()
# Nova coluna de total de Gols somando os gols do time da casa com os do time Visitante

matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']
# 5 jogos que tiveram maior quantidade de gols

matches.nlargest(5, 'Total Goals').T
# 5 jogos que tiveram menor quantidade de gols

matches.nlargest(5, 'Total Goals').T
# 3 jogos com maior publico

matches.nlargest(3, 'Attendance').T
#Fazendo graficos usando o seaborn

import seaborn as sns

import matplotlib.pyplot as plt
#Boxplot do total de gols por ano

# Usando o metplotlib que pode desenhar em uma área maior



plt.figure(figsize=(15,5))

sns.boxplot(matches['Year'], matches['Total Goals'])

plt.title('Gols por Ano')

plt.xticks(rotation=65)

plt.locator_params(axis='y',nbins=20)

plt.show()
wcup.info()
# Gols marcados por edição da Copa do Mundo



plt.figure(figsize=(15,5))

sns.pointplot(x="Year", y='GoalsScored', data=wcup, color = 'green')

plt.title('Total de Gols marcados por Copa do Mundo')

plt.grid(True, color='grey')
#Calcular o total de gols marcados de cada seleção em todas as Copas do Mundo



# Primeiro como mandante



gols_mandante = matches.groupby('Home Team Name')['Home Team Goals'].sum().reset_index()

gols_mandante.columns = ['Team', 'Goals']

gols_mandante.head()


# segundo como visitante



gols_visitante = matches.groupby('Away Team Name')['Away Team Goals'].sum().reset_index()

gols_visitante.columns = ['Team', 'Goals']

gols_visitante.head()
# Juntando os dataframes

total_gols = pd.concat([gols_mandante, gols_visitante], axis=0)



total_gols.sample(10)
# Conferindo gols da Argentina

total_gols[total_gols['Team']=='Argentina']
# Conferindo gols da Brazil

total_gols[total_gols['Team']=='Brazil']
# Agrupar o dataframe de total



total_gols = total_gols.groupby("Team")["Goals"].sum().reset_index()



total_gols[total_gols['Team']=='Brazil']
total_gols.head()
total_gols.info()

#Transformando em inteiro

total_gols['Goals'] = total_gols["Goals"].astype(int)
#Ordenando o dataframe

total_gols = total_gols.sort_values(by='Goals', ascending = False)



#plotando o grafico de barras

plt.figure(figsize=(15,5))

sns.barplot(x='Team', y='Goals', data=total_gols[:10])

plt.locator_params(axis='y',nbins=20)

plt.show()
# Outra forma de somar os dados

top_gols = pd.DataFrame(matches.groupby('Home Team Name')['Home Team Goals'].sum()

                       +

                       matches.groupby('Away Team Name')['Away Team Goals'].sum())

top_gols.nlargest(10,0)