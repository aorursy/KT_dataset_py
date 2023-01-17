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
#Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates
from datetime import date, timedelta
import statsmodels.api as sm
import statsmodels.api as tsa
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARMA
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
#Importar e apresentar os dados
casos = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
casos.head()
#Informações das variáveis
casos.info()
#Quantidade de missing values
casos.isna().sum()
#Renomear colunas
casos = casos.rename(columns={"ObservationDate":"date", "Country/Region": "country", "Province/State": "state","Confirmed":"confirm", "Deaths": "death","Recovered":"recover"})
casos.head()
#Contagem de casos
casos[casos['date'] != casos['Last Update']]['country'].value_counts()
#Remover colunas
#casos.drop(columns=['SNo', 'state'], inplace=True, axis=1)
#casos.head()
casos1 = casos[casos['country']=='Brazil']

casos1.head()
#Numero de casos maximos em 1 dia
casos1['confirm'].idxmax()
#remover valores 0 da variável 'confirm' 
zeroConfirmed = casos1[casos1['confirm'] == 0]
casos1 = casos1[casos1['confirm'] != 0]
casos1.shape
#casos = casos.groupby("country")[['confirm', 'death', 'recover']].sum().reset_index()
#casos.head()
#Grafico com a quantidade casos confirmados, recuperados e de mortos por Covid-19.

plt.figure(figsize=(23,10))
plt.bar(casos1.date, casos1.confirm,label="Confirmados")
plt.bar(casos1.date, casos1.recover,label="Recuperados")
plt.bar(casos1.date, casos1.death,label="Mortos")
plt.xlabel('Data')
plt.ylabel("Contagem")
plt.legend(frameon=True, fontsize=10)
plt.title("Casos confirmados, Recuperados e Mortos",fontsize=30)
plt.xticks(rotation='vertical')
plt.show()

ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="date", y="confirm", data=casos1,
             color="black",label = "Confirmados")
ax=sns.scatterplot(x="date", y="recover", data=casos1,
             color="red",label = "Recuperados")
ax=sns.scatterplot(x="date", y="death", data=casos1,
             color="blue",label = "Mortos")
plt.plot(casos1.date,casos1.confirm,zorder=1,color="black")
plt.plot(casos1.date,casos1.recover,zorder=1,color="red")
plt.plot(casos1.date,casos1.death,zorder=1,color="blue")
plt.legend(frameon=True, fontsize=10)
plt.title("Casos confirmados, Recuperados e Mortos",fontsize=20)
plt.xticks(rotation='vertical')
plt.show()
#Filtrar as colunas 
casos1.drop(columns=['SNo', 'state', 'death', 'recover', 'Last Update', 'country'], inplace=True, axis=1)
casos1.head()
#Indexando a coluna date para criar a série temporal.

casos1.index = casos1['date']
casos1.drop('date',inplace=True, axis=1)
casos1.head()
#Gráfico com o volume de casos confirmados e evolução do tempo.
plt.figure(figsize=(15,6))
plt.plot(casos1)
plt.xlabel('Data')
plt.ylabel('Casos confirmados')
plt.title('Data vs casos confirmados')
plt.xticks(rotation='vertical')
plt.show()
mean = casos1.rolling(window=12).mean()
m_std = casos1.rolling(window=12).std()

ori = plt.plot(casos1, color='blue', label='Original')
mm = plt.plot(mean, color='red',label='Média')
std = plt.plot(m_std, color='black', label='Desvio')
plt.legend(loc='best')
plt.title('Evolução dos casos de coronavírus')
plt.xticks(rotation='vertical')
plt.show()

result = adfuller(casos1['confirm'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
#Variação de casos confirmados durante o periodo de Fevereiro a Abril 
per = np.log(casos1).diff().dropna()
per.plot()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(casos1)
casos1.shift(1)
modelo = ARIMA(casos1, order=(0, 1, 1)).fit()
print(modelo.summary())
#Previsão do Corona Vírus
fig, ax = plt.subplots(figsize=(8,6))
fig = modelo.plot_predict(start='02-27-2020', end='04-18-2020', ax=ax)
plt.title('Previsão Corona Vírus')
plt.ylabel('Confirmados')
plt.xlabel('Data')
legend = ax.legend(loc='upper left')
#Modelo Naive sem sazonalidade
naive = ARIMA(casos1, order=(0, 0, 0))
naive = naive.fit()
print(naive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,4)})
naive_treinado.plot_predict(start=1,end=220)
plt.show()
#Modelo naive com sazonalidade

snaive = ARIMA(casos1, order=(0, 1, 0))
snaive = snaive.fit()
print(snaive_treinado.summary())
#Treinando o modelo com sazonalidade
snaive.plot_predict(start=1,end=230)
plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Fit the model
mod = sm.tsa.statespace.SARIMAX(casos1, trend='c', order=(1,1,1))
res = mod.fit(disp=False)
print(res.summary())
fig ,ax= plt.subplots(2,1, figsize=(10,7))
fig=sm.tsa.graphics.plot_acf(casos1, lags=50, ax=ax[0])
fig=sm.tsa.graphics.plot_pacf(casos1, lags=50, ax=ax[1])
plt.show()
mod_sarimax = SARIMAX(casos1, order=(1,1,1), seazonal_order=(1,1,1,4))
mod_sarimax.fit()
plt.show()