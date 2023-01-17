# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importação dos arquivos usados na análise

worldCup = pd.read_csv('../input/WorldCups.csv', encoding='utf-8')

matches = pd.read_csv('../input/WorldCupMatches.csv', encoding='utf-8')

players = pd.read_csv('../input/WorldCupPlayers.csv', encoding='utf-8')
# Dimensões das bases de dados

print('Copa do Mundo: ', worldCup.shape)

print('Copa do Mundo: ', matches.shape)

print('Copa do Mundo: ', players.shape)
worldCup.info()
players.info()
matches.info()
worldCup.head()
matches.head()
players.head()
worldCup.columns
matches.columns
players.columns
# Criação de coluna com o total de gols em uma determinada partida

matches['TotalGols'] = matches['Home Team Goals'] + matches['Away Team Goals']
matches.head()
# Ajuste do campo Attendance para o tipo inteiro

worldCup['Attendance'] = worldCup['Attendance'].str.replace('.','').astype(int)
worldCup.info()
worldCup.describe()
players.describe()
matches.describe()
matches.nlargest(5, 'Attendance')
matches.nsmallest(5, 'Attendance')
matches.nlargest(5, 'TotalGols')
# Maiores campeões

plt.figure(figsize=(15,5))

sns.countplot(x='Winner', data=worldCup, order=worldCup['Winner'].value_counts().index)

plt.title("Maiores vencedores",color='black')

plt.show()
# Seleções participantes

plt.figure(figsize=(15,4))

sns.barplot(x='Year',y='QualifiedTeams', data=worldCup)

plt.title("Quantidade de seleções participantes",color='black')

plt.show()
# Quantidade de partidas disputadas por copa do mundo

plt.figure(figsize=(15,4))

sns.pointplot(x='Year', y ='MatchesPlayed', data=worldCup, color="orange")

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de partidas por copa do mundo",color='black')

plt.show()
# Média de gols por copa do mundo

plt.figure(figsize=(15,5))

sns.boxplot(matches['Year'], matches['TotalGols'])

plt.title("Média de gols",color='black')

plt.xticks(rotation=90)

plt.show()
# Quantidade de gols marcados por copa do mundo

plt.figure(figsize=(15,5))

sns.pointplot(x='Year', y ='GoalsScored', data=worldCup, color="green")

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de gols marcados por copa do mundo",color='black')

plt.show()
# Média de público por copa do mundo

plt.figure(figsize=(15,5))

sns.boxplot(matches['Year'], matches['Attendance'])

plt.title("Público médio por partida",color='black')

plt.xticks(rotation=90)

plt.show()
# Público absoluto por copa do mundo

plt.figure(figsize=(15,4))

sns.pointplot(x='Year', y ='Attendance', data=worldCup, color="red")

plt.grid(True,color="grey",alpha=.3)

plt.title("Público por copa do mundo",color='black')

plt.show()
# Distribuição dos gols por partida

plt.figure(figsize=(15,4))

sns.distplot(matches["TotalGols"].dropna(),rug=True, color='purple')

plt.xticks(np.arange(0,9,1))

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos gols por partida",color='black')

plt.show()

# Total de gols por seleção

gols_casa = matches.groupby("Home Team Name")["Home Team Goals"].sum().reset_index()

gols_casa.columns = ["selecao","gols"]

gols_fora = matches.groupby("Away Team Name")["Away Team Goals"].sum().reset_index()

gols_fora.columns = ["selecao","gols"]

gols = pd.concat([gols_casa, gols_fora],axis=0)

gols = gols.groupby("selecao")["gols"].sum().reset_index()

gols = gols.sort_values(by="gols",ascending =False)

gols["gols"] = gols["gols"].astype(int)



plt.figure(figsize=(15,5))

sns.barplot(x="selecao",y="gols", data=gols[:10])

plt.title("Seleções com maior número de gols marcados",color='black')

plt.show()