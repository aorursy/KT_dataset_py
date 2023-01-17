# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Importa as bibliotecas de gráficos

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Criando o dataframe



fifa2019 = pd.read_csv('../input/data.csv')



# Verificando informação do banco da dados

fifa2019.info()

# Visualizando os 10 prmeiros registros



fifa2019.head()
# Visualiznado os dados aleatoriamente

fifa2019.sample(20)
# Verificando a quantidade de nulos

fifa2019.isnull().sum()
# Estatisticas basicas dos dados 

fifa2019.describe()
# Alterando a quantidade de colunas

novas_colunas = ['Acceleration', 'Age', 'Aggression', 'Agility', 'Balance', 'BallControl',

                 'Body Type', 'Club', 'Composure', 'Contract Valid Until', 'Crossing', 'Curve',

                 'Dribbling', 'Finishing', 'FKAccuracy', 'Flag', 'GKDiving', 'GKHandling', 'GKKicking',

                 'GKPositioning', 'GKReflexe', 'GKReflexes', 'HeadingAccuracy', 'Height', 'Interceptions',

                 'International Reputation', 'Jersey Number', 'Joined', 'Jumping', 'LongPassing', 'LongShots',

                 'Marking', 'Name', 'Nationality', 'Overall', 'Penalties', 'Position', 'Positioning', 'Potential',

                 'Preferred Foot', 'Reactions', 'ShortPassing', 'ShotPower', 'SlidingTackle', 'Special',

                 'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Value', 'Vision', 'Volleys', 'Wage',

                 'Weak Foot', 'Weight', 'Work Rate']



fifa_novo = pd.DataFrame(fifa2019 , columns = novas_colunas)

fifa_novo.tail()
# Correlação das variaveis

plt.rcParams['figure.figsize'] = (25,20)

mapa_calor = sns.heatmap(fifa_novo[['Age', 'Overall', 'Potential', 'Value', 'Acceleration', 'Aggression', 

                'Agility', 'Balance', 'BallControl', 'Body Type','Composure', 'Crossing',

                'Dribbling', 'FKAccuracy', 'Finishing', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots', 'Marking', 'Penalties', 

                'Position', 'Positioning', 'ShortPassing', 'ShotPower', 'SprintSpeed', 'Stamina', 

                'StandingTackle', 'Strength', 'Vision']].corr(), annot = True, linewidths = .5, cmap = 'Reds')

mapa_calor.set_title(label = 'Mapa de Calor', fontsize = 20);
# Histograma de número de jogadores em relação a idade



sns.set(style ="dark") 

plt.figure(figsize=(15,8))

ax = sns.distplot(fifa_novo.Age, bins = 58, kde = False)

ax.set_xlabel(xlabel="Idade dos jogadores", fontsize=16)

ax.set_ylabel(ylabel='Número de jogadores', fontsize=16)

ax.set_title(label='Histograma Idade/Quantidade', fontsize=20)

plt.show()
# Número de jogadores por país

plt.figure(1 , figsize = (35 , 15))

plt.xticks(rotation = 90)

plt.title('Número de jogadores por país')

sns.countplot(x = "Nationality", data = fifa_novo, palette="rocket" )

# O melhor jogador por posição

top_player = fifa_novo.iloc[fifa_novo.groupby(fifa_novo['Position'])['Overall'].idxmax()][['Name', 'Position','Club']]

top_player.set_index('Name', inplace=True)

top_player

# Distribuição dos salarios com base na idade e desempenho dos jogadores

plt.figure(figsize = (15,10))

sns.scatterplot(data = fifa_novo, y = 'Wage', x = 'Overall', hue = 'Age', size = 'Age', sizes = (20, 200), palette = "Set1")

plt.title("Distribuição de salários com base na idade dos jogadores e no desempenho geral")

plt.xlabel("Pontuação")

plt.ylabel("Salário (Em Euros)")
# Preferência de chutes dos jogadores



EsqDir = fifa_novo['Preferred Foot'].value_counts()

plt_fifa_novo = [go.Bar(

    x = EsqDir.index,

    y = EsqDir

    )]

layout = go.Layout(

    autosize=False,

    width=500,

    height=500,

    title = "Preferência de chute dos jogadores"

)

figura = go.Figure(data = plt_fifa_novo, layout=layout)

iplot(figura)
# 30 jogadores mais velhos

top_old = fifa_novo.sort_values('Age' , ascending = False)[['Name' , 'Club' , 'Age' , 'Overall']].head(30)

top_old
# 10 jogadores mais novos

top_young = fifa_novo.sort_values('Age' , ascending = True)[['Name' , 'Club' , 'Age' , 'Overall']].head(10)

top_young
# 10 melhores jogadores no geral

best_player = fifa_novo.sort_values('Overall' , ascending = 0)[['Overall' , 'Name' , 'Value' , 'Club' , 'Position']].head(10)

best_player.set_index('Name' , inplace = True)

best_player
# Média de reputação internacional por idade

plt.figure(figsize=(10,5))

idade = fifa_novo.sort_values("Age")['Age'].unique()

reputacao = fifa_novo.groupby(by="Age")["International Reputation"].mean().values

plt.title("Idade X Reputação Internacional ")

plt.xlabel("Idade")

plt.ylabel("Reputação Internacional")

plt.plot(idade, reputacao)

plt.show()
# Gráfico de scatter plot entre controle de bola e passe longo

fifa_scatter = fifa_novo.loc[:100,:]

fifa_scatter.plot(kind="scatter",x="BallControl",y="LongPassing",c="g",linewidth=0.4,grid=True,alpha=0.4, figsize=(10,5))

plt.xlabel('Controle de Bola', fontsize = 14)

plt.ylabel('Passe Longo', fontsize = 14)

plt.show()
# Boxplot para testar se a idade interfere na carreira

plt.figure(figsize = (15,10))

sns.boxplot(x = "Work Rate", y = "Age", data = fifa_novo)

plt.show()