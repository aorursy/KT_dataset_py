from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import warnings
warnings.filterwarnings("ignore")
## Lendo o arquivo

bova11 = pd.read_csv('/kaggle/input/bova-11/BOVA11.SA.csv')
## Tornando o nome das colunas mais "pythonico"

bova11.columns = 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'
## Transformando a coluna date em tipo data 

bova11['date'] = pd.DatetimeIndex(bova11['date'])
## Adicionando colunas de ano, mes e semana

bova11['year'] = bova11['date'].apply(lambda x: x.year)

bova11['month'] = bova11['date'].apply(lambda x: x.month)

bova11['week'] = bova11['date'].apply(lambda x: x.week)
bova11.info()
bova11 = bova11.dropna()
bova11.describe()
plt.figure(figsize=[20, 7])

plt.title('Cotação maxima por dia da BOVA11 ao longo do tempo')

plt.plot(bova11[bova11['date'].between('2014-01-01', '2014-12-31')]['high'], label='2014')

plt.plot(bova11[bova11['date'].between('2015-01-01', '2015-12-31')]['high'], label='2015')

plt.plot(bova11[bova11['date'].between('2016-01-01', '2016-12-31')]['high'], label='2016')

plt.plot(bova11[bova11['date'].between('2017-01-01', '2017-12-31')]['high'], label='2017')

plt.plot(bova11[bova11['date'].between('2018-01-01', '2018-12-31')]['high'], label='2018')

plt.plot(bova11[bova11['date'].between('2019-01-01', '2019-12-31')]['high'], label='2019')

plt.legend()

plt.show()
X = bova11[bova11['year'] >= 2016]['high']
plt.figure(figsize=[20, 7])

plt.title('Cotação maxima por dia da BOVA11 ao longo do tempo a partir de 2016')

plt.plot(X)

plt.show()
plt.figure(figsize=[20, 7])

plt.title('Distribuição dos valores maximos por dia da BOVA11')

plt.hist(X)

plt.show()
## O Teste de Dickey-Fuller serve para verificar se uma serie temporal é estacionaria.

def dickey_fuller(serie):

    resultado = adfuller(serie)

    print('Estatistica ADF: %f' % resultado[0])

    print('p-valor: %f' % resultado[1])

    print('Valores criticos:')

    for chave, valor in resultado[4].items():

        print('\t%s: %.3f' % (chave, valor))
dickey_fuller(X)
serie_estacionaria = X - X.shift(-2)

serie_estacionaria = serie_estacionaria.dropna()
dickey_fuller(serie_estacionaria)



plt.figure(figsize=[20, 7])

plt.title('BOVA11 apos o processo de se tornar estacionaria')

plt.plot(serie_estacionaria, label='Valores')



plt.legend()

plt.show()



plt.figure(figsize=[20, 7])

plt.title('Distribuicao dos valores da BOVA11 após se tornar estacionaria')

plt.hist(serie_estacionaria)

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))



plot_acf(serie_estacionaria, ax=ax1)

plot_pacf(serie_estacionaria, ax=ax2)

plt.show()
tamanho_teste = 60

tamanho_treino = len(X) - tamanho_teste

treino = list(X[0:tamanho_treino])

teste = list(X[tamanho_treino:])



ordem = (1, 2, 0)
modelo = ARIMA(treino, order=ordem).fit()
fig, axs = plt.subplots(2, 2, figsize=(20, 7))



print('Media: %.2f' % np.mean(modelo.resid))

print('Desvio: %.2f\n' % np.std(modelo.resid))



dickey_fuller(modelo.resid)



axs[0][0].hist(modelo.resid)

axs[0][1].plot(modelo.resid)



plot_acf(modelo.resid, ax=axs[1][0])

plot_pacf(modelo.resid, ax=axs[1][1])



plt.show()
previsao_diario = []

for i in range(0, tamanho_teste):

    modelo_diario = ARIMA(list(X[0:tamanho_treino + i]), order=ordem).fit()

    previsao_diario.append(modelo_diario.forecast()[0][0])
rmse_diario = np.sqrt(np.sum(np.power(np.subtract(teste, previsao_diario), 2)))

print('RMSE: %.2f' % rmse_diario)



plt.figure(figsize=[20, 7])

plt.plot(teste, label='Real')

plt.plot(previsao_diario, label='Previsto')



plt.xticks([x for x in range(0, tamanho_teste)])

plt.grid(axis='x')

plt.legend()

plt.show()