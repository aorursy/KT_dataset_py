import numpy as np # álgebra linear
import pandas as pd # entrada de dados
import seaborn as sns # plotagem de gráficos
import pylab as pl #para mudar tamanho de figuras

import os

data = pd.read_csv('../input/avocado-prices/avocado.csv')

print('Dados inicializados.')
#Verifica o tamanho
data.shape
#Imprimindo os dados (10 primeiras linhas)
data.head(10)
#Verifica se existem células não preenchidas
data.isnull().sum()
#Plotando um gráfico de distribuição do tipo KDE (Kernel Density Estimate)
pl.figure(figsize=(10,5))
pl.title("Distribuição tipo KDE - Preços dos abacates")

sns.kdeplot(data.AveragePrice)
#Gráfico boxplot dos preços de abacate: conventional x organic
pl.figure(figsize=(18,5))
pl.title("Preços de abacates - geral")

sns.boxplot(
    x='AveragePrice',
    data=data
)

#Imprimindo a média dos valores
data["AveragePrice"].mean()
#Gráfico boxplot dos preços de abacate: conventional x organic
pl.figure(figsize=(18,5))
pl.title("Preços de abacates - comparação entre tipos")

sns.boxplot(
    x='AveragePrice',
    y='type',
    data=data
)

#Obtendo os valores das médias de abacate
data.groupby('type').AveragePrice.mean()
#Gráfico boxplot dos preços de abacate: conventional x organic
pl.figure(figsize=(18,20))
pl.title("Preços de abacates - comparação entre regiões")

sns.boxplot(
    x='AveragePrice',
    y='region',
    data=data
)

#Abacates mais caros e baratos
print('------ REGIOES COM ABACATES MAIS CAROS ------')

print('\n>>> Geral:\n')
caro1 = data.groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro1)

print('\n>>> Convencionais:\n')
caro2 = data.loc[data['type'] == 'conventional'].groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro2)

print('\n>>> Orgânicos:\n')
caro3 = data.loc[data['type'] == 'organic'].groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro3)

#Abacates mais caros e baratos
print('\n------ REGIOES COM ABACATES MAIS BARATOS ------')

print('\n>>> Geral:\n')
barato1 = data.groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato1)

print('\n>>> Convencionais:\n')
barato2 = data.loc[data['type'] == 'conventional'].groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato2)

print('\n>>> Orgânicos:\n')
barato3 = data.loc[data['type'] == 'organic'].groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato3)
#Gráfico boxplot dos preços de abacate: conventional x organic
pl.figure(figsize=(18,20))
pl.title("Preços de abacates - comparação entre regiões - 2018")

dataSoh2018 = data.loc[data['year'] == 2018]

sns.boxplot(
    x='AveragePrice',
    y='region',
    data= dataSoh2018 # Restringe para as entradas com ano de 2018.
)

#Abacates mais caros e baratos
print('------ REGIOES COM ABACATES MAIS CAROS ------')

print('\n>>> Geral:\n')
caro1 = dataSoh2018.groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro1)

print('\n>>> Convencionais:\n')
caro2 = dataSoh2018.loc[data['type'] == 'conventional'].groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro2)

print('\n>>> Orgânicos:\n')
caro3 = dataSoh2018.loc[data['type'] == 'organic'].groupby(['region']).AveragePrice.mean().nlargest(3)
print(caro3)

#Abacates mais caros e baratos
print('\n------ REGIOES COM ABACATES MAIS BARATOS ------')

print('\n>>> Geral:\n')
barato1 = dataSoh2018.groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato1)

print('\n>>> Convencionais:\n')
barato2 = dataSoh2018.loc[data['type'] == 'conventional'].groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato2)

print('\n>>> Orgânicos:\n')
barato3 = dataSoh2018.loc[data['type'] == 'organic'].groupby(['region']).AveragePrice.mean().nsmallest(3)
print(barato3)
#Gráfico heatmap
pl.figure(figsize=(15,15))
pl.title("Comparação entre variáveis")

colunas = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
dataCorrigida = np.corrcoef(data[colunas].values.T) # Corrigindo valores 
sns.heatmap(dataCorrigida, annot = True, square = True, annot_kws = {'size':15}, yticklabels = colunas, xticklabels = colunas, cbar = False)
from IPython.display import Image

Image("../input/imagem-correlao-pearson/correlacao-pearson.jpg")
#Gráfico jointplot entre "Total Volume" e "Total Bags"

print("Imprimindo o coeficiente de correlação...")
print(data['Total Bags'].corr(data['Total Volume']))

dataVolBags = data.loc[data['Total Bags'] < 20000]

sns.jointplot(x='Total Volume', y='Total Bags', data=dataVolBags, kind='hex', gridsize = 15)
#Gráfico jointplot entre "Small Bags" e "Large Bags"

print("Imprimindo o coeficiente de correlação...")
print(data['Small Bags'].corr(data['Large Bags']))

dataSmallLarge = data.loc[data['Small Bags'] < 20000].loc[data['Large Bags'] < 20000]

sns.jointplot(x='Large Bags', y='Small Bags', data=dataSmallLarge, kind='hex', gridsize = 15)
#Gráfico jointplot entre "Year" e "Average Price"

print("Imprimindo o coeficiente de correlação...")
print(data['year'].corr(data['AveragePrice']))

dataYearPrice = data.loc[data['AveragePrice'] < 5]

sns.jointplot(x='AveragePrice', y='year', data=dataYearPrice, kind='hex', gridsize = 15)
#Gráficos boxplot em relação a anos e tipos
pl.figure(figsize=(5, 5))
pl.title("Boxplot em relação a anos e tipos - convencional")

sns.boxplot(
    x='year',
    y='AveragePrice',
    data= data.loc[data['type'] == "conventional"]
)

anosOrganicos = data.loc[data['type'] == 'organic'].groupby(['year']).AveragePrice.mean()
anosConvencionais = data.loc[data['type'] == 'conventional'].groupby(['year']).AveragePrice.mean()

print("Médias Orgânicos:\n")
print(anosOrganicos)

print("\n\nMédias convencionais:\n")
print(anosConvencionais)


#####################################################3

pl.figure(figsize=(5, 5))
pl.title("Boxplot em relação a anos e tipos - orgânico")

sns.boxplot(
    x='year',
    y='AveragePrice',
    data= data.loc[data['type'] == "organic"]
)
