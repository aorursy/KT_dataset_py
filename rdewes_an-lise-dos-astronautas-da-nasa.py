# Importando bibliotecas



import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
# Local dos arquivos



INPUT = '../input'

import os

print(os.listdir(INPUT))
# Carga dos dados

astronautas = pd.read_csv(INPUT+'/astronauts.csv')
# Análise inicial da base

print('Forma da tabela:', astronautas.shape)

print('Estrutura do dataframe:')

print(astronautas.info())
# correção das datas

astronautas['Birth Date'] = pd.to_datetime(astronautas['Birth Date'])

astronautas['Death Date'] = pd.to_datetime(astronautas['Death Date'])
# Verificação dos dados

astronautas.sample(5)
# Dados das variáveis numéricas

astronautas.describe()



# Verifica-se que existem diversos NaN devido a diferença na contagem das variáveis
# Análise das categóricas

astronautas['Military Branch'].value_counts()

# Removido o tipo (Retired) e reagrupado.

astronautas['tipo_milico'] = astronautas['Military Branch'].str.replace(' \(Retired\)', '')

astronautas['tipo_milico'].value_counts()
# Rank Militar

astronautas['Military Rank'].value_counts()
# Generos

astronautas['Gender'].value_counts(normalize=True) * 100



# 86% dos astronautas são homens
# Top 10 formações

astronautas['Undergraduate Major'].value_counts().head(10)



# Existem astronautas com mais de uma formação
# Top 10 Universidades

astronautas['Alma Mater'].value_counts().head(10)



## Diversos astronautas tem mais de uma universidade de formação
# Verificando que astronautas não tem ano preenchido

astronautas[astronautas.Year.isnull()].head(10)
# Verificando a distribuição das variáveis

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



## Quantidades de voos

sns.distplot(astronautas['Space Flights'], ax=axs[0])

sns.boxplot(astronautas['Space Flights'], ax=axs[1])
## Horas de voos

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



sns.distplot(astronautas['Space Flight (hr)'], ax=axs[0])

sns.boxplot(astronautas['Space Flight (hr)'], ax=axs[1])
## Camihadas Espaciais

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



sns.distplot(astronautas['Space Walks'], ax=axs[0])

sns.boxplot(astronautas['Space Walks'], ax=axs[1])
## Camihadas Espaciais Horas

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



sns.distplot(astronautas['Space Walks (hr)'], ax=axs[0])

sns.boxplot(astronautas['Space Walks (hr)'], ax=axs[1])
# mortes por ano

astronautas['ano_morte'] = astronautas['Death Date'].dt.year

astronautas['ano_morte'].value_counts().to_frame().reset_index().sort_values('index').plot.bar(y='ano_morte', 

                                                                                               x='index', label='mortes por ano', rot=45)



# O ano com mais mortes é 1986
## Quais os 5 astronautas com mais horas de voos?

astronautas[['Name', 'Space Flight (hr)', 'Space Flights']].sort_values('Space Flight (hr)', ascending=False).head()
## E os 5 astronautas com mais caminhadas de voos?

astronautas[['Name', 'Space Walks (hr)', 'Space Walks']].sort_values('Space Walks (hr)', ascending=False).head()
## Missões que falharam (morreram astronautas)

astronautas[astronautas['Death Mission'].notnull()]['Death Mission'].unique()
# Média das missões por genero

astronautas[['Gender', 'Space Flights']].groupby('Gender').mean()
# Média das missões por status

astronautas[['Status', 'Space Flights']].groupby('Status').mean()
# Verificando se Mission e Space Flights são em mesma quantidade

astronautas['x'] = astronautas['Missions'].str.split(',')  # Missoes são separadas por virgula

astronautas['qtd_missoes'] = astronautas[astronautas['x'].notnull()]['x'].apply(lambda x: len(x))  # Conta tamanho da lista

# astronautas.sort_values('qtd_missoes', ascending=False).head()

astronautas[(astronautas['Space Flights']!=astronautas['qtd_missoes'])&(astronautas['qtd_missoes'].notnull())]



# Existem quatro linhas em que Space Flights difere de quantidade missões
# Vendo se o povo aposentado tem mais horas de espaço que a ativa

astronautas[['Status', 'Space Flight (hr)']].groupby('Status').sum()