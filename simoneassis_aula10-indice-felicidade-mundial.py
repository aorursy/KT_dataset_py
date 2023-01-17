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
#Carregando os dados
dados_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
dados_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
dados_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

#Olhando o tamnho dos dataframes
print('2015: ',dados_2015.shape)
print('2016: ',dados_2016.shape)
print('2017: ',dados_2017.shape)
# Dados de 2015
dados_2015.head()
#Dados de 2016
dados_2016.head()
#Dados de 2017
dados_2017.head()
#últimos por ano
dados_2015.tail()
#dados_2016.tail()
#dados_2017.tail()
#filtrando por Brasil
dados_2015[dados_2015['Country']=='Brazil']
#Mostrando os 5 menores
#nlargest 5 maiores
#nsmallest 5 maiores
dados_2015.nlargest(5,'Happiness Rank')
#Qual a posiçaõ do Brasil em cada ano?
dados_2015[dados_2015['Country']=='Brazil']['Happiness Rank']
dados_2016[dados_2016['Country']=='Brazil']['Happiness Rank']

dados_2017[dados_2017['Country']=='Brazil']['Happiness.Rank']
novo =pd.merge(dados_2015, dados_2017, on= 'Country',sort=True)
new =pd.merge(novo, dados_2017, on= 'Country',sort=True)
new.head().T
dados_2016[dados_2016['Country']=='Afghanistan']
#Juntando os dataframes
#Código exemplo: result = pd.merge(left,right, on='key')
df_resultado = pd.merge(dados_2015,dados_2016, on='Country')
df_resultado.head().T
#Continuamos Juntando os dataframes
df_resultado = pd.merge(df_resultado,dados_2017, on='Country')
df_resultado.head().T
#posição do Brasil
df_resultado[df_resultado['Country'] == 'Brazil'].T
#Dados de 2015
#Criando um coluna para o ano
dados_2015['ano'] = 2015
dados_2015.head()
#importando bibliotecas
import matplotlib.pyplot as plt
import seaborn as sns
#Relacionamento happiness score e PIB
dados_2015.plot(title = '2015', kind='scatter', x='Economy (GDP per Capita)',y='Happiness Score',color='red')

dados_2016.plot(title = '2016', kind='scatter', x='Economy (GDP per Capita)',y='Happiness Score',color='green')

dados_2017.plot(title = '2017', kind='scatter', x='Economy..GDP.per.Capita.',y='Happiness.Score',color='blue')
dados_2017.head().T
#Gráfico de happiness por região
sns.stripplot(x='Region',y='Happiness Score',data=dados_2015)
plt.xticks(rotation=90)
#Plotando a correlação

# Aumentando a área do gráfico

f,ax=plt.subplots(figsize=(15,6))
sns.heatmap(dados_2015.corr(), annot=True, fmt='.2f', linecolor='black', ax=ax, lw=.7)
#criando a coluna happy_qualitiy
#com base nos quartis de happiness score

q3=dados_2016['Happiness Score'].quantile(0.75)
q2=dados_2016['Happiness Score'].quantile(0.5)
q1=dados_2016['Happiness Score'].quantile(0.25)

happy_quality= []

#Percorrer a coluna do dataframe e determinar as categorias
for valor in dados_2016['Happiness Score']:
    if valor >= q3:
        happy_quality.append('Muito Alto')
    elif valor >= q2 and valor < q3:
         happy_quality.append('Alto')
    elif valor >= q1 and valor < q2:
         happy_quality.append('Normal')    
    else: 
         happy_quality.append('Muito Baixo')
            
dados_2016['happy_quality'] = happy_quality


dados_2016.head(5)
#Gráfico usando o happy_quality

plt.figure(figsize=(7,7))
sns.boxplot(dados_2016['happy_quality'], dados_2016['Economy (GDP per Capita)'])
plt.figure(figsize=(10,7))
sns.swarmplot(dados_2016['happy_quality'], dados_2016['Economy (GDP per Capita)'])
#Correlação Health (Life Expectancy) com Economy (GDP per capita)
#Categorização por happy_quality


sns.scatterplot (dados_2016['Health (Life Expectancy)'], dados_2016['Economy (GDP per Capita)'], 
                 hue=dados_2016['happy_quality'], style=dados_2016['happy_quality'])