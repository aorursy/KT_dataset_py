
#Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



#importing data
data = pd.read_csv('../input/videogamesales/vgsales.csv')
# 5 first lines
data.head()
# simple info about data
data.info()
# Looking values nan
data.isna().sum()
# Looking column "Year"

data['Year']


# converte os valores, Float64, da coluna "Year" para datetime 
def to_formatData(x):
    if(not np.isnan(x)):
        x = int(x)
        date = pd.to_datetime(x, yearfirst=True, format="%Y")
        return date

# criando nova coluna "year" com valores de "Year" convertidos para datetime
data['year'] = data['Year'].apply(to_formatData)

# new data
data.head()

# Imprimindo o primeiro e último ano de registro (intervalo de anos)

print('Data mínima: ', data['year'].min())
print('Data maxíma: ', data['year'].max())

# verificando coluna "Name"

#data['Name'].unique().shape
data['Name'].value_counts()
data[data['Name'] == data['Name'].value_counts().index[0]] # imprimindo linhas em que "Name" = "Need for Speed: Most Wanted", para verificar os registros com mesmo "Name" 

data['Platform'].value_counts().plot.bar(figsize=(12,3))
g = plt.gca()
g.set_title('Count Plataform rank: 1980 até 2020', size=14)
g.set_ylabel('Count', size=14)
g.set_xlabel('Platform', size=14)
plt.show()

data['Publisher'].value_counts(normalize=True)[0:30].plot.bar(figsize=(12,3)) # 30 maiores publicadoras dos games
g = plt.gca()
g.set_title('Count greather Publisher: 1980 até 2020', size=14)
g.set_xlabel('Publisher', size=14)
g.set_ylabel('Frequency', size=14)
plt.show()

data['Genre'].value_counts().plot.bar()
g = plt.gca()
g.set_title('Count Genre title')
g.set_xlabel('Genre')
g.set_ylabel('Count')
plt.show()


data2 = data.dropna() # apagando registros com Nan
data2 = data2.set_index(data2['year']);  # colocando index como a coluna 'year'
data2.drop(['Year','year'], inplace=True, axis=1) # deletando colunas 'Year' e 'year'
#data2.index.isna().sum()
data2.info()
data2.head()

# realizando uma soma de todos os registros que apresentam o mesmo ano, dessa forma
# é possível verificar o total das vendas 'Global_Sales' por cada ano, abaixo é realiado uma plotagem
# do compartamento de 'Global_Sales' durante os anos.

globalSalesPerYear = data2['Global_Sales'].resample('Y').sum()
globalSalesPerYear
globalSalesPerYear.plot(figsize=(12,4))
g = plt.gca()
g.set_ylabel('Total Global Sales')
g.set_title('Total Global Sales per year')
plt.show()
data2[data2.index >= pd.to_datetime('2017')] # pegando todos os registros a partir do ano de 2017

# considerando o ano de 2010, por exemplo

#m = data2[data2.index == pd.to_datetime('2006-01-01')]
#m[m['Global_Sales'] == m['Global_Sales'].max()]


# Intervalo de anos dentro dos dados. freq = 'YS' significa frequencia em anos, pegano o inicio do ano
# 'yearStart'. Experimente usar apenas freq = 'Y' para ver o resultado. Mais sobre esses detalhes em [2] e [3]
dateRange = pd.date_range('1980-01-01','2020-01-01', freq='YS'); #dateRange

# Títulos com maior venda em cada ano

MVendaPerYear = pd.DataFrame(columns=data2.columns)

for y in dateRange:
    temp = data2[data2.index == y] # seleciona os registros por ano
    MVendaPerYear = pd.concat([MVendaPerYear, temp[temp['Global_Sales'] == temp['Global_Sales'].max()]])
    
MVendaPerYear

# resumindo dados acima
sns.catplot(x='Name', y='Global_Sales', data=MVendaPerYear, height=5, aspect=4, kind='bar')
g = plt.gca()
g.set_title('Títulos com maiores vendas de cada ano (1980 - 2020)', size=14)
plt.xticks(rotation=90)
plt.show()
sns.catplot(x='Genre', y='Global_Sales', data=data2, height=4, aspect=3)
g = plt.gca()
g.set_title('Global Sales per Genre')
sns.catplot(x='Platform', y='Global_Sales', data=data2, height=4, aspect=3)
g = plt.gca()
g.set_title('Global Sales per Platform')
