#Importando e instalando as bibliotecas necessárias

!pip install pandas-profiling

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pandas_profiling

import seaborn as sns

%matplotlib inline
#Carregando o dataset

df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

df.head()
df.info()
#Convertendo os tipos de dados

df.year = df.year.astype('int32')

df.state = df.state.astype('category')

df.month = df.month.astype('category')

df.number = df.number.astype('float64') 
df.info()
#Analizando os dados gerais do dataframe

pandas_profiling.ProfileReport(df)
#Agrupando os estados e calculando a média de casos

df.groupby('state')['number'].mean().nlargest(27)
#Criando uma dataframe com a junção dos valores de 'year', 'state' e 'month', fazendo isso é preciso resetar o index.

year_state_month = df.groupby(by = ['year','state', 'month']).sum().reset_index()
year_state_month.head()
plt.figure(figsize=(12,4))



sns.boxplot(x = 'state', order = ['Sao Paulo', 'Mato Grosso', 'Bahia','Piau','Maranhao','Amazonas','Pará','Tocantins','Rondonia','Acre'], 

            y = 'number', data = year_state_month)



plt.title('BoxPlot Incêncio por Estado', fontsize = 18)

plt.xlabel('Estado', fontsize = 14)

plt.ylabel('Número de casos', fontsize = 14)
#Número de incêndios por ano

fires_year = df.groupby('year')['number'].sum().reset_index()

fires_year.groupby
sns.set_style('whitegrid')



from matplotlib.pyplot import MaxNLocator



plt.figure(figsize=(12,4))



ax = sns.lineplot(x = 'year', y = 'number', data = year_state_month, estimator = 'sum', color = 'purple', lw = 2, 

                  err_style = None)



plt.title('Total de incêndios no Brasil : 1998 - 2017', fontsize = 18)

plt.xlabel('Ano', fontsize = 14)

plt.ylabel('Número de incêndios', fontsize = 14)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)