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
#carregando a base em dataframe



bike_london = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv')

#Verificando os tamanhos ds dataframes

print('bike compartilhadas:', bike_london.shape)
#Visualisando os dados carregados

bike_london.head(10)
#Verificando dados nulos na base

bike_london.isnull().any()
#Visualisando as colunas carregadas no dataFrame

bike_london.info()

# Converção em Data campo Timestamp

bike_london['timestamp']=pd.to_datetime(bike_london['timestamp'])

bike_london.info()
#Criando coluna Data, Hora, Va

bike_london['data']=pd.to_datetime(bike_london['timestamp'].dt.strftime('%Y-%m-%d'))

bike_london['hora']=pd.to_timedelta(bike_london['timestamp'].dt.strftime('%H:%M:%S'))

bike_london['ano'] =pd.DatetimeIndex(bike_london['timestamp']).year

bike_london['mes_valor'] = pd.DatetimeIndex(bike_london['timestamp']).month

bike_london['mes_ano']=bike_london['timestamp'].dt.strftime('%b-%Y')



bike_london.info()
#Analise Descritiva da Base -  Visualização dos Dados do dataFrame bike_london após a inclusão das colunas

bike_london.head()
#Analise Descritiva da Base - ALUGUÉIS totais POR ANO_mes

bike_london.groupby(['mes_ano']).sum()['cnt'].T
# #Analise Descritiva da Base  - 10 menores quantidades de bicicletas alugadas

bike_london.nsmallest(10,'cnt')
#Analise Descritiva da Base  - 10 maiores quantidades de bicicletas alugadas

bike_london.nlargest(10,'cnt')

#Analise Descritiva da Base  -  histograma da quantidade de bicicletas alugadas

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=[7,5])

plt.hist(bike_london['cnt'], color='red', bins=6)

plt.title('Histograma da quantidade de bicicletas alugadas')

plt.show()
#Analise Descritiva da Base - Boxblot do total de Bicicletas alugadas  ano

plt.figure(figsize=(10,15))

sns.boxplot(bike_london['ano'], bike_london['cnt'])

plt.title('Bicicletas alugadas  ano')

#plt.xticks(rotation=65)

plt.locator_params(axis='y',nbins=3)

plt.show()
#Quantidade de Bicicletas Alugadas por Ano

bike_london_ano = bike_london.groupby('ano').agg('sum')



#reset_index

bike_london_ano.reset_index(inplace=True)

#Criando um dataframe com as tres primeiras colunas

bike_london_ano=bike_london_ano.iloc[:, 0:2].copy()

bike_london_ano.head()
#Grafico com o aluguel por ano



#Aumentando a área do grafico

fig, ax=plt.subplots(figsize=(14,7))



#Criadno um grafico de barras com o seaborn

sns.barplot(y='cnt',x='ano', data=bike_london_ano )



#Titulos

plt.title('Quantidade de Bicicletas alugadas por mês', fontsize=24)

plt.ylabel('Quantidade',fontsize=24)

plt.xlabel('Ano',fontsize=24)
#Quantidade de Bicicletas Alugadas por Mês

bike_london_mes = bike_london.groupby('mes_ano').agg('sum')

#reset_index

bike_london_mes.reset_index(inplace=True)

#Criando um dataframe com as tres primeiras colunas

bike_london_mes=bike_london_mes.iloc[:, 0:2].copy()

bike_london_mes.head()
#Analise Descritiva da Base  -  Bicicletas alugadas por mes ano

plt.figure(figsize=(30,5))

sns.pointplot(x='mes_ano',y='cnt',data = bike_london_mes, color='blue',)

plt.title('Quantidade de Bicicletas alugadas por mês')

plt.grid(True, color='grey')
#Analise Descritiva da Base  - Correlação entre Bicicletas alugadas e temperatura ambiente e sensação termica



plt.figure(figsize=[20,10])

plt.title('Correlação entre a Quantidade de Bicicletas alugadas por temperatura ambiente')

sns.scatterplot(bike_london['t1'],bike_london['cnt'],

                style=bike_london['ano'],hue=bike_london['ano']

               )

plt.figure(figsize=[20,10])

plt.title('Correlação entre a Quantidade de Bicicletas alugadas por Sensação Termica')

sns.scatterplot(bike_london['t1'],bike_london['cnt'],

                style=bike_london['ano'],hue=bike_london['ano']

               )
# Verificando correlação das variaveis da base

bike_london.corr()
#modelo de regressão multipla 

import statsmodels.formula.api as sm

reg = sm.ols(formula='cnt~t1+t2+hum+wind_speed+weather_code+is_holiday+is_weekend+season+ano+mes_valor', data=bike_london).fit()

print(reg.summary())

#Analise dos Residuos do modelo

cnt_pre = reg.predict()

res = bike_london['cnt'] - cnt_pre



plt.hist(res, color='orange', bins=15)

plt.title('Histograma dos resíduos da regressão')

plt.show()
plt.scatter(y=res, x= cnt_pre, color='green', s=50, alpha=.6)

plt.hlines(y=0, xmin=-10, xmax=3000, color='orange')

plt.ylabel('$\epsilon = cnt - \hat{y}$ - Resíduos')

plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')

plt.show()