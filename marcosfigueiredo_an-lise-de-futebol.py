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
# Importando mais Bibliotecas

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sbn

import datetime as dt

import re
# Importando Bases de Dados de Jogadores Inscritos na Fifa em 2019

data = pd.read_csv("../input/data.csv")
# Conferindo as primeiras linhas da Base de Dados

data.head()
# Verificando Colunas da Base de Dados

data.columns
# Excluindo algumas colunas

data.drop(['Unnamed: 0','Photo','Flag','Club Logo'], axis = 1, inplace = True)
# Verificação de Campos com valores Nulos

data.isnull().sum()
# Transformando o Salário dos Jogadores em campo numérico

data['Salario'] = (data['Wage'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64'))
# Conferindo Novamente as primeiras linhas da Base de Dados

data.head()
# Contando a quantidade de Países em nossa Base de Dados

data.Nationality.nunique()
# Agora vamos identificar os 10 países que tem a maior quantidade de jogadores

data['Nationality'].value_counts().head(10)
# Também é possível identificarmos a quantidade de Jogadores em cada Clube

data['Club'].value_counts().head(10)
# Identificando a média de Idade dos Jogadores

data.Age.mean()
# Identificando a média de Idade dos Jogadores por Clube

Club_Id = data.groupby('Club').Age.mean()

print(Club_Id)
# Identificando Quantos Jogadores em cada Idade há

data['Age'].value_counts()
# Identificando a média de Salário dos Jogadores

data.Salario.mean()
# Levantamento Estatístico da Base de Dados

data.describe().T
# Podemos então verificar quais são os melhores Jogadores por Caracterítica como Chute ou Cabeceio

# Ou mesmo por Posição quem é o melhor Goleiro ou Atacante

# A exemplo disso vamos trazer quais os melhores finalizadores



finaliza = data.sort_values(by='Finishing',ascending = False )

finaliza.head()
# Gráfico de Barras de Idade de Jogadores

sbn.set(rc={'figure.figsize':(12,12)})

data.Age.plot(kind='hist', bins=20)
# Gráfico de Barras com as Notas Gerais dos Jogadores 

data.Overall.plot(kind='hist', bins=20)
#Listando os Clubes da Base de Dados

print(data.Club)
# Gerando BoxPlot de Salários para os Clubes Extrangeiros mais comentados atualmente no Brasil

TopClubs = data[(data.Club == 'FC Barcelona') | (data.Club == 'Juventus') | (data.Club == 'Real Madrid') | (data.Club == 'FC Bayern Munich') | (data.Club == 'Paris Saint-Germain') ]

sbn.set(style="whitegrid", color_codes=True)

sbn.boxplot(x="Club", y="Salario", hue="Club", data=TopClubs, palette="PRGn")

plt.title('BOXPLOT SALÁRIOS', fontsize=25)

sbn.despine(offset=10, trim=True)
# Através do Gráfico foi possível identificar Salários muito altos, OutLiers

# assim para vamos identificar quais clubes têm a maior quantidade de atletas

# com slários elevados.

# Após alguns testes quanto a apresentação do Gráfico, defino como OutLier

# todos com Overall a partir de 85.



Outlier = data[data['Overall']>= 85]

Agrup = Outlier.groupby('Club')

Count = Agrup.count()['Name'].sort_values(ascending = False)

ax = sbn.countplot(x = 'Club', data = Outlier, order = Count.index)

ax.set_xticklabels(labels = Count.index, rotation='vertical')

ax.set_ylabel('Quantidade de Jogadores')

ax.set_xlabel('Clube')

ax.set_title('Clubes e seus Melhores Jogadores', fontsize=25)
# Gerando matriz de correlação

Analise = data[['Age','Overall','Potential','Salario','Agility','Finishing','Acceleration','BallControl','FKAccuracy','Jumping','LongPassing','LongShots','Dribbling']]
# Criando e Plotando a matriz de correlação

Correlacao = Analise.corr()

ax = sbn.heatmap(Correlacao, xticklabels=Correlacao.columns.values, yticklabels=Correlacao.columns.values,

linewidths=1.0, vmax=1.0, square=True, cmap = 'PuBu', linecolor='white', annot=False)