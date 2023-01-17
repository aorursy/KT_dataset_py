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
#Importando as bibliotecas



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm

import statsmodels.api as tsa

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.stattools import adfuller, arma_order_select_ic

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6



from matplotlib import pyplot as plt



from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



from random import random

#from pandas.plotting import register_matplotlib_converters

%matplotlib inline
#Importar a base Brasil

Brasil = pd.read_excel('/kaggle/input/Brasil.xlsx', skiprows=12, usecols=[0,1,4], encoding='latin1')



#Apresentar os dados Brasil

Brasil.head()
#Renomear colunas

Brasil.rename(columns={"MÊS": "Periodo", "PRODUTO":"Combustíveis", "PRECO MÉDIO REVENDA": "Preços"}, inplace=True)

Brasil.head()
#Apresentando informações das variaveis 

Brasil.info()
#Filtrar a coluna Combustiveis

Brasil = Brasil[Brasil['Combustíveis'] == 'ETANOL HIDRATADO']

Brasil.head()
#Remover a coluna combustiveis

Brasil.drop(columns=['Combustíveis'], inplace=True, axis=1)

Brasil.head()
#Importar o dataset Mensal_Brasil configurando apresentação dos dados. 

mensalB = pd.read_excel('/kaggle/input/MENSAL_BRASIL-DESDE_Jan2013.xlsx',skiprows=15,usecols=[0,1,4],encoding='latin1')

mensalB.head()
#Renomear colunas

mensalB.rename(columns={"MÊS":"Periodo","PRODUTO":"combustiveis","PREÇO MÉDIO REVENDA":"Preços"}, inplace=True)

mensalB.head()
# Apresentando o tipo de dados na base.

mensalB.info()
#filtrar a coluna Combustiveis

mensalB = mensalB[mensalB['combustiveis'] =='ETANOL HIDRATADO']

mensalB.head()
#Remover a coluna Combustiveis

mensalB.drop(columns=['combustiveis'], inplace=True, axis=1)

mensalB.head()
#Unir os datasets apos ajustes

dados = pd.concat([Brasil, mensalB], ignore_index=True, sort=True)

dados.head()
#atribuindo a coluna periodo no index na base

dados.index = dados['Periodo']

dados.drop('Periodo',inplace=True, axis=1)

dados.head()
plt.plot(dados)

plt.title('Evolução do preço do Etanol Hidratado')

plt.xlabel('Periodo')

plt.ylabel('Preços')

plt.show()
#Plotar gráfico de Periodo e preços



X = [i for i in range(0, len(dados))]

X = np.reshape(X, (len(X), 1))

y = dados

modelo = LinearRegression()

modelo.fit(X, y)



modelo.coef_

tendencia = modelo.predict(X)



plt.plot(dados.values, color='blue',label='Original')

plt.plot(tendencia,color='red', label='Tendência' )

plt.title('Regressão Periodo por preço')

plt.xlabel('Periodo')

plt.ylabel('Preços')

plt.show()
mean = dados.rolling(window=12).mean()

m_std = dados.rolling(window=12).std()



ori = plt.plot(dados, color='blue', label='Original')

mm = plt.plot(mean, color='red',label='Média')

std = plt.plot(m_std, color='black', label='Desvio')

plt.legend(loc='best')

plt.title('Evolução do preço do Etanol Hidratado')



plt.show()



result = adfuller(dados['Preços'])

print('ADF Statistic: {}'.format(result[0]))

print('p-value: {}'.format(result[1]))

print('Critical Values:')

for key, value in result[4].items():

    print('\t{}: {}'.format(key, value))
#Estatistica do modelo

arma_mod30 = sm.tsa.ARMA(dados, (3,0)).fit(disp=False)



print(arma_mod30.summary())
#Previssão do preço para 12 meses

fig, ax = plt.subplots(figsize=(8,6))

fig = arma_mod30.plot_predict(start='2001-07-01', end='2020-10-01', ax=ax)

plt.title('Previssão do preço para 12 meses, utilizando o modelo ARMA')

plt.ylabel('Preços')

plt.xlabel('Periodo')

legend = ax.legend(loc='upper left')
#Variação de periodo e preço

per = np.log(dados).diff().dropna()

per.plot()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(dados)



#plotar o gráfico de autocorrelação dos resíduos entre duas séries

dados.shift(1)
dados_diff = dados.diff(periods=1) 

#Integrar e ordenar
dados_diff = dados_diff[1:]

dados_diff.head()
#Plotar o gráfico de autocorrelação 

plot_acf(dados_diff)
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(dados.values.squeeze(), lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(dados, lags=40, ax=ax2)
dados_diff.plot()
modelo = ARIMA(dados, order=(0, 1, 1)).fit()

print(modelo.summary())
#Previssão do preço 12 meses

fig, ax = plt.subplots(figsize=(8,6))

fig = modelo.plot_predict(start='2001-08-01', end='2020-10-01', ax=ax)

plt.title('Previssão do preço 12 meses, modelo ARIMA')

plt.ylabel('Preços')

plt.xlabel('Periodo')

legend = ax.legend(loc='upper left')
#Previssão do preço para 12 meses

fig, ax = plt.subplots(figsize=(8,6))

fig = modelo.plot_predict(start='2001-08-01', end='2020-10-01', ax=ax)

plt.title('Previssão do preço para 12 meses, utilizando o modelo ARIMA')

plt.ylabel('Preços')

plt.xlabel('Periodo')

legend = ax.legend(loc='upper left')
#Modelo Naive sem sazonalidade

naive = ARIMA(dados, order=(0, 0, 0))

naive_treinado = naive.fit()

print(naive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,3)})

naive_treinado.plot_predict(start=1,end=230)

plt.show()
#Modelo naive com sazonalidade



snaive = ARIMA(dados, order=(0, 1, 0))

snaive_treinado = snaive.fit()

print(snaive_treinado.summary())
#Treinando o modelo com sazonalidade

snaive_treinado.plot_predict(start=1,end=230)

plt.show()
 #sem Tendencia e Ajuste sazonal

model = ExponentialSmoothing(dados)

hw_model = model.fit()

pred = hw_model.predict(start=dados.index[0], end=dados.index[219])

plt.title('Holt-Winters Filtering')

plt.plot(dados.index, dados, label='Original', color='k')

plt.plot(pred.index, pred, label='Holt-Winters',color='r')

plt.legend(loc='best');
#Modelo com tendencia e ajuste sazonal



model = ExponentialSmoothing(dados, trend='mul',seasonal='mul')

hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)

pred = hw_model.predict(start=14, end=219)

plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')

plt.plot(dados.index, dados, label='Original')

plt.plot(pred.index, pred, label='Holt-Winters')

plt.legend(loc='best');

plt.show()
#Modelo Holt-Winters com Tendencia e Ajuste sazonal, previsão para 36 meses



model = ExponentialSmoothing(dados, seasonal_periods=7 ,trend='add', seasonal='mul',damped=True,)

hw_model = model.fit()

pred = hw_model.predict(start='2019-11-01', end='2022-10-01')

plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')

plt.plot(dados.index, dados, label='Original')

plt.plot(pred.index, pred, label='Holt-Winters')

plt.legend(loc='best');

plt.show()