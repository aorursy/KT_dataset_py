# código responsável por gerar os dados de decomposição dos casos confirmados de covid-19 na cidade de blumenau

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose

date_parse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('blumenau.csv', parse_dates= ['Date'], index_col= 'Date', date_parser= date_parse)

data = df.Confirmed # podemos mudar essa coluna para Death, UTI ou Recovered e gerar decomposição para qualquer um desses dados
data = data.dropna().asfreq('d')

decomposition = seasonal_decompose(data)
trend = decomposition.trend
seasonal = decomposition.seasonal
resid = decomposition.resid

plt.subplot(4, 1, 1)
plt.plot(data, label = 'Original')
plt.legend(loc = 'best')

plt.subplot(4, 1, 2)
plt.plot(trend, label = 'Tendencia')
plt.legend(loc = 'best')

plt.subplot(4, 1, 3)
plt.plot(seasonal, label = 'Sazonalidade')
plt.legend(loc = 'best')

plt.subplot(4, 1, 4)
plt.plot(resid, label = 'Aleatorio')
plt.legend(loc = 'best')

plt.tight_layout()
plt.show()

import pandas as pd
import pmdarima as pm
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA

date_parse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('blumenau.csv', parse_dates= ['Date'], index_col= 'Date', date_parser= date_parse)

data = df.Confirmed
data = data.dropna().asfreq('d')

modelo_auto = pm.auto_arima(data, m=30)

modelo = ARIMA(data, order=modelo_auto.get_params()['order'])
modelo_treinado = modelo.fit()
modelo_treinado.summary()


eixo = data.plot()
modelo_treinado.plot_predict('2020-07-01', '2020-07-31', ax = eixo, plot_insample = True)
plt.show()


