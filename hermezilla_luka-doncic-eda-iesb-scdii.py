# Importando as bibliotecas

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#criando dataset

nba = pd.read_csv('/kaggle/input/luka-doncic-stats/NBA matches.csv')

nba.head(5).T

#Percebemos como existem colunas sem nome ou que não refletem o atributo

#Vamos tratá-las a seguir
#Ajustando o cabeçalho dos dados - pegando os nomes das colunas

colunas = list(nba.columns)

colunas
#Ajustando o cabeçalho dos dados - Colocando nomes significativos

colunas[0] = 'GameID'

colunas[1] = 'GamesPlayed'

colunas[5] = 'LocalVisitor'

colunas[6] = 'Opponent'

colunas[7] = 'WinLoss'

colunas[8] = 'Started'

colunas[9] = 'MIN'

colunas[28] = 'GameScore'

colunas[29] = 'PlusMinus'



nba.columns = colunas

nba.tail(10).T
# Ao analisar o fim da tabela, percebemos em primeiro lugar que só tem informações de 75 jogos.

# A temporada da NBA tem 82 jogos, dessa forma, o dataset encontra-se incompleto.

# Muito provavelmente foi disponibilizado antes da temporada regular acabar.

print(nba.shape)

# Desses 75 jogos, 5 Luka Doncic não participou. Estamos estudando o jogador, então essas

# informações serão retiradas do dataframe

nba = nba[nba['GamesPlayed'].notnull()]

print(nba.shape)
# Vamos verificar os tipos dos dados

nba.info()
# Tratando a data

nba['Date'] = pd.to_datetime(nba['Date'])
# A informação de idade conta os anos e dias, mas torna a coluna dificil de lidar.

# Não me interessa saber os dias exatos então vamos transformar o campo.

nba['Age'] = pd.to_numeric(nba['Age'].str[:2])
# A informação de vitória ou derrota consta também a diferença de pontos.

# É uma informação valiosa, mas não na mesma coluna. Vamos separá-las

nba['Difference'] = pd.to_numeric(nba['WinLoss'].str[2:8].str.replace('(','').str.replace(')','').str.replace('+',''))

nba['WinLoss'] = nba['WinLoss'].str[:1]
# Tratando minutos, não importa a parte fracionária

nba['MIN'] = pd.to_numeric(nba['MIN'].str[:2])
# Variaveis numéricas

nba['FG'] = pd.to_numeric(nba['FG'])

nba['FGA'] = pd.to_numeric(nba['FGA'])

nba['FG%'] = pd.to_numeric(nba['FG%'])

nba['3P'] = pd.to_numeric(nba['3P'])

nba['3PA'] = pd.to_numeric(nba['3PA'])

nba['3P%'] = pd.to_numeric(nba['3P%'])

nba['FT'] = pd.to_numeric(nba['FT'])

nba['FTA'] = pd.to_numeric(nba['FTA'])

nba['FT%'] = pd.to_numeric(nba['FT%'])

nba['ORB'] = pd.to_numeric(nba['ORB'])

nba['DRB'] = pd.to_numeric(nba['DRB'])

nba['TRB'] = pd.to_numeric(nba['TRB'])

nba['AST'] = pd.to_numeric(nba['AST'])

nba['STL'] = pd.to_numeric(nba['STL'])

nba['BLK'] = pd.to_numeric(nba['BLK'])

nba['TOV'] = pd.to_numeric(nba['TOV'])

nba['PF'] = pd.to_numeric(nba['PF'])

nba['PTS'] = pd.to_numeric(nba['PTS'])

nba['GameScore'] = pd.to_numeric(nba['GameScore'])

nba['PlusMinus'] = pd.to_numeric(nba['PlusMinus'].str.replace('+',""))
#Verificando o banco novamente

nba.info()
# Exploração básica 

nba.describe().T
# Nos jogos em que jogou pelo menos 35 minutos, quantos ganhou ou perdeu?

nba[nba['MIN'] >= 35]['WinLoss'].value_counts()
# Nos jogos em que fez pelo menos 30 pontos, quantos ganhou ou perdeu?

nba[nba['PTS'] >= 30]['WinLoss'].value_counts()
# Nos jogos em que seu saldo de pontos foi positivo enquanto esteve em quadra

# (plusMinus), quantos ganhou ou perdeu?

nba[nba['PlusMinus'] > 0]['WinLoss'].value_counts()
# Nos jogos em que seu time jogou em casa, quantos ganhou ou perdeu?

nba[nba['LocalVisitor'] != '@']['WinLoss'].value_counts()
#Distribuição de +/- nos jogos em que ganhou

sns.distplot(nba[nba['WinLoss'] == 'W']['PlusMinus'])
# Lances livres convertidos em relação aos lances livres tentados

sns.lineplot(x='FTA',y='FT',data=nba)
# Gráfico de vioino da distribuição dos pontos nos jogos que ganhou ou perdeu por idade

sns.set(style="whitegrid", palette="pastel", color_codes=True)

sns.violinplot(x="Age", y="PTS", hue="WinLoss",

               split=True, inner="quart",

               palette={"W": "y", "L": "b"},

               data=nba)

sns.despine(left=True)
# Gráfico de dispersão de pontos por minutos jogados levando em consideração a quantidade de bolas de 3 e se venceu ou não

sns.set()

sns.scatterplot(x="MIN", y="PTS",

                     hue="WinLoss", size="3P",

                     sizes=(10, 200),

                     data=nba)
#Jogos vencidos com 20 anos

nba[(nba['WinLoss'] == 'W') & (nba['Age'] == 20)]