import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import pandas as pd

import statsmodels.api as sm

import matplotlib



from sklearn.metrics import mean_squared_error

from math import sqrt

# Usei desta forma no Windows 10

#import os

#os.chdir('C:\\Users\\rgosd\\Google Drive\\000hotlearn\\serpro-kaggle\\serpro-weather')

#wseries = pd.read_csv("input/weather-train.csv", header=0, index_col=0, names=['date', 'temperature'], parse_dates=True, squeeze=True)



# importar e visualizar base de dados

prefixo_arquivos = '../input/serpro-weather/'

wseries = pd.read_csv(prefixo_arquivos + 'weather-train.csv', header=0, index_col=0, names=['date', 'temperature'], parse_dates=True, squeeze=True)
print(wseries.describe())

print(str(wseries.size))

print(wseries.head())
wseries[wseries.isnull()]
wseries.index.get_loc('2013-03-01')
wseries = wseries.interpolate()

wseries.iloc[wseries.index.get_loc('2013-03-01')]
# definindo valores default de plotagem

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'



warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
wseries.plot(figsize=(15, 4))

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 18, 9



decomposition = sm.tsa.seasonal_decompose(wseries, model='additive', freq=365)

fig = decomposition.plot()

plt.show()
wseries.columns = ['temperature']
wseries.head()
# obs: ocorreram alguns erros na instalação do fbprophet em windows 10, usei dicas da primeira resposta de https://stackoverflow.com/questions/53178281/installing-fbprophet-python-on-windows-10
from fbprophet import Prophet

from fbprophet.plot import plot_plotly

# lendo novamente o csv para facilitar colocar no formato prophet

df = pd.read_csv(prefixo_arquivos + "weather-train.csv")

df['ds'] = df['date']

df['y'] = df['temperature']

df.drop(['date','temperature'],axis = 1, inplace = True)

df.head()
m = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1)

m.fit(df)

# usei para previsão o tamanho do test set: 378 dias

future = m.make_future_dataframe(periods=378)

forecast = m.predict(future)

m.plot(forecast);

m.plot_components(forecast);
forecast.tail()

# usando opção de plotagem dos changepoints

from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
# Arbitrado aqui tamanho do conjunto de testes = tamanho a ser enviado (75%)



# tamanho inicial: 1508

# tamanho do test: 378

# tamanho do treino = 1508 - 378 = 1130

train_size = 1130



train, test = df[0:train_size], df[train_size:]

print(train.shape)

print(train.head())

print(train.tail())

print(test.shape)

print(test.head())

print(test.tail())
m = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.04, n_changepoints=4)

m.fit(train)

future = m.make_future_dataframe(periods=378)

forecast = m.predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
m.plot_components(forecast);
predictions = list()

testset = list()

# faz a previsão

yhat = forecast['yhat']

#predições: seleciona o conjunto além do treino

predictions.append(yhat[train_size:])

#seleciona os valores conhecidos

testset.append(test['y'])

rmse = sqrt(mean_squared_error(testset, predictions))

print(rmse)
# Agora usa o conjunto de treino total para gerar o modelo final

mfinal = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.04, n_changepoints=2)

mfinal.fit(df)

future = mfinal.make_future_dataframe(periods=378)

forecast = mfinal.predict(future)

fc = forecast[['ds','yhat']]

fc.columns = ['date', 'temperature']

fc.set_index('date')

submission_test = fc[1508:]

print(submission_test.head())

submission_test.shape

submission_test.to_csv('RG_serpro_weather_01v2_2cp.csv', index=False)