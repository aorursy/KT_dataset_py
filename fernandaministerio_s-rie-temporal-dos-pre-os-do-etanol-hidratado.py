import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from random import random
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statsmodels.api as sm
# %matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importando os dataset df e df2

df = pd.read_excel('/kaggle/input/brasil/Brasil.xlsx',skiprows=12,usecols=[0,1,4],encoding='latin1')
df2 = pd.read_excel('/kaggle/input/mensal/MENSAL_BRASIL-DESDE_Jan2013.xlsx',skiprows=15,usecols=[0,1,4],encoding='latin1')
rawData = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',skiprows=12,usecols=[0,1,4],encoding='latin1')
rawData.head()
#Renomeando colunas do dataset
rawData.rename(columns={"MÊS": "periodo", "PRODUTO": "combustiveis","PRECO MÉDIO REVENDA": "precos"},inplace=True)
rawData.info()
# Filtrando o dataset pela colunas combustiveis
parsedData = rawData[rawData['combustiveis'] =='ETANOL HIDRATADO']
# removendo a coluna combustiveis
parsedData = parsedData.drop(columns=['combustiveis'])
parsedData.head()
#  o segundo dataset e fazendo o primeiro filtro nas linhas e colunas
rawData2 = pd.read_excel('../input/srie-temporal-etanol-hidratado/MENSAL_BRASIL-DESDE_Jan2013.xlsx',skiprows=15,usecols=[0,1,4],encoding='latin1')
rawData2.head()
# Renomeando as colunas do segundo dataset
rawData2.rename(columns={"MÊS": "periodo", "PRODUTO": "combustiveis","PREÇO MÉDIO REVENDA": "precos"},inplace=True)
rawData2.head()
rawData2.info()
parsedData2 = rawData2[rawData2['combustiveis'] =='ETANOL HIDRATADO']
parsedData2 = parsedData2.drop(columns=['combustiveis'])
parsedData2.head()
dados = pd.concat([
    parsedData,parsedData2
],ignore_index=True,sort=True)
dados.count()
#atribuindo a coluna periodo no index da base
dados.index = dados['periodo']
dados.drop('periodo',inplace=True, axis=1)
dados.head()
plt.plot(dados)
plt.title('Evolução do preço do Etanol Hidratado')
plt.xlabel('Periodo')
plt.ylabel('Preços')
plt.show()
X = [i for i in range(0, len(dados))]
X = np.reshape(X, (len(X), 1))
y = dados
modelo = LinearRegression()
modelo.fit(X, y)
modelo.coef_
tendencia = modelo.predict(X)

plt.plot(dados.values, color='blue',label='Original')
plt.title('Evolução do preço do Etanol Hidratado')
plt.ylabel('Periodo')
plt.xlabel('Preços')
plt.legend('Y = 0.00028*x-2.27836')
plt.plot(tendencia,color='red', label='Tendência' )
plt.show()
# Estatísticas do modelo ARMA, Tem que adicionar gráficos 
modelo_arma = sm.tsa.ARMA(dados, (3,0)).fit(disp=False)
print(modelo_arma.summary())
#Previssão do preço para 12 meses
fig, ax = plt.subplots(figsize=(8,6))
fig = modelo_arma.plot_predict(start='2001-07-01', end='2020-10-01', ax=ax)
plt.title('Previssão do preço para 12 meses, utilizando o modelo ARMA')
plt.ylabel('Preços')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
# Estatísticas do modelo ARIMA, Falta adicionar gráficos 
modelo_arima = ARIMA(dados, order=(0, 1, 1)).fit()
print(modelo_arima.summary())
#Previssão do preço para 12 meses
fig, ax = plt.subplots(figsize=(8,6))
fig = modelo_arima.plot_predict(start='2001-08-01', end='2020-10-01', ax=ax)
plt.title('Previssão do preço para 12 meses, utilizando o modelo ARIMA')
plt.ylabel('Preços')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')

naive = ARIMA(dados, order=(0, 0, 0))
naive_treinado = naive.fit()
print(naive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,3)})
naive_treinado.plot_predict(start=1,end=230)
plt.show()

snaive = ARIMA(dados, order=(0, 1, 0))
snaive_treinado = snaive.fit()
print(snaive_treinado.summary())
snaive_treinado.plot_predict(start=1,end=230)
plt.show()
model = ExponentialSmoothing(dados)
hw_model = model.fit()
pred = hw_model.predict(start=dados.index[0], end=dados.index[219])
plt.title('Holt-Winters Filtering')
plt.plot(dados.index, dados, label='Original', color='k')
plt.plot(pred.index, pred, label='Holt-Winters',color='r')
plt.legend(loc='best');

model = ExponentialSmoothing(dados, seasonal_periods=36)
hw_model = model.fit()
pred = hw_model.predict(start='2019-11-01', end='2022-10-01')
plt.plot(dados.index, dados, label='Original')
plt.title('Forecasts from HoltWinters')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()
model = ExponentialSmoothing(dados, trend='add',seasonal='None')
hw_model = model.fit()
pred = hw_model.predict(start=0, end=219)
plt.title('Holt-Winters Filtering')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()
model = ExponentialSmoothing(dados, trend='mul',seasonal='mul')
hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
pred = hw_model.predict(start=14, end=219)
plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()

#model = ExponentialSmoothing(dados, trend='None', seasonal='None', seasonal_periods=36, damped=True)
#hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
#pred = hw_model.predict(start=test.index[0], end=test.index[-1])
model = ExponentialSmoothing(dados, seasonal_periods=7 ,trend='add', seasonal='mul',damped=True,)
hw_model = model.fit()
pred = hw_model.predict(start='2019-11-01', end='2022-10-01')
plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()