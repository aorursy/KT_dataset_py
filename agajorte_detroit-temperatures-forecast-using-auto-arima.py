# importar pacotes necessários

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# instalar pacotes especiais

!pip install pmdarima
# definir parâmetros extras

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
# definir frequência a ser considerada no modelo

frequencia = '2W' # 7D 15D 2W 3W M

periodos_ano = 26 # 365.25 / 14



# definir data inicial de corte

data_inicio_amostra = '2013-01'
# calcular períodos que cabem em um ano

#intervalo_ano = pd.date_range(start='2018-01-01', end='2019-01-01', freq=frequencia)

#periodos_ano = len(intervalo_ano)

print('Frequência:', frequencia)

print('Períodos em um ano:', periodos_ano)

print('Data de início da amostra:', data_inicio_amostra)
prefixo_arquivos = '/kaggle/input/detroit-daily-temperatures-with-artificial-warming/'
# carregar arquivo de dados de treino

all_data = pd.read_csv(prefixo_arquivos + 'weather-complete.csv', index_col='date', parse_dates=['date'])

all_data.info()

all_data.head()
# remover valores nulos

all_data.dropna(inplace=True)



# reduzir a quantidade de dados para a frequência

data = all_data.resample(frequencia).mean()



# filtrar período desejado

data = data[data_inicio_amostra:]

#data = data['2013-01':]

#data = data['2013-01':'2015-12']



# converter temperatura para Kelvin

#data['temperature'] += 273.15



data.head()
data.info()
data.describe()
# criar série temporal a partir do dataframe

ts = data['temperature']

ts = ts.asfreq(frequencia)

ts.head()
# dividir dados entre treino e teste

corte = (len(ts) - 1 * periodos_ano) # 1 ano

treino = ts[:corte]

teste = ts[corte:]

print('Treino:', treino.shape)

print('Teste: ', teste.shape)
# register datetime converter for a matplotlib plotting method

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# plotar a série temporal

plt.plot(treino)

plt.title('Temperatura ao longo dos anos (em graus Celsius)', fontsize=20)

plt.show()
from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries, window):

    

    # Determing rolling statistics

    rolmean = timeseries.rolling(window).mean()

    rolstd = timeseries.rolling(window).std()



    # Plot rolling statistics

    orig = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label='Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation', fontsize=20)

    plt.show(block=False)

    

    # Perform Dickey-Fuller test

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries)

    dfoutput = pd.Series(dftest[0:4], index=[

        'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)
# avaliar se a série é estacionária

test_stationarity(treino, periodos_ano)
# Plot

fig, axes = plt.subplots(2, 1, figsize=(14,8), dpi=100, sharex=True)



# Usual Differencing

axes[0].plot(treino, label='Original Series')

axes[0].plot(treino.diff(1), label='Usual Differencing')

axes[0].set_title('Usual Differencing')

axes[0].legend(loc='upper left', fontsize=10)



# Seasonal 1st

axes[1].plot(treino, label='Original Series')

axes[1].plot(treino.diff(periodos_ano), label='Seasonal Differencing', color='green')

axes[1].set_title('Seasonal Differencing')

plt.legend(loc='upper left', fontsize=10)

plt.suptitle('Temperaturas', fontsize=16)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose



ts_log = treino

decomposition = seasonal_decompose(ts_log, freq=periodos_ano)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(ts_log, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Tendência')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Sazonalidade')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Resíduos')

plt.legend(loc='best')

plt.tight_layout()
# https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html



from pmdarima.arima.utils import ndiffs



dft = pd.DataFrame({

    'Teste': [

        'ADF (Augmented Dickey-Fuller)',

        'KPSS (Kwiatkowski–Phillips–Schmidt–Shin)',

        'PP (Phillips–Perron)'

    ],

    'Valor estimado para o termo "d"': [

        ndiffs(treino, test='adf'),

        ndiffs(treino, test='kpss'),

        ndiffs(treino, test='pp')

    ]

})

dft.set_index('Teste', inplace=True)

dft
# https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.nsdiffs.html



from pmdarima.arima.utils import nsdiffs



dft = pd.DataFrame({

    'Teste': [

        'OCSB (Osborn-Chui-Smith-Birchenhall)',

        'CH (Canova-Hansen)'

    ],

    'Valor estimado para o termo "D"': [

        nsdiffs(treino, periodos_ano, test='ocsb'),

        nsdiffs(treino, periodos_ano, test='ch')

    ]

})

dft.set_index('Teste', inplace=True)

dft
# https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html



from pmdarima import auto_arima



smodel = auto_arima(treino, start_p=1, start_q=1,

                         test='adf',

                         max_p=3, max_q=3, m=periodos_ano,

                         start_P=0, seasonal=True,

                         d=None, D=1, trace=True,

                         error_action='ignore',  

                         suppress_warnings=True, 

                         stepwise=True)



smodel.summary()
# realizar a previsão

periodos = teste.shape[0] + 1

fitted, confint = smodel.predict(n_periods=periodos, return_conf_int=True)

index_of_fc = pd.date_range(teste.index[0], periods=periodos, freq=frequencia)



# criar séries para plotagem

fitted_series = pd.Series(fitted, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)



# plotar gráfico

plt.plot(treino)

plt.plot(fitted_series, color='darkgreen')

plt.fill_between(lower_series.index, 

                 lower_series, 

                 upper_series, 

                 color='k', alpha=.15)

plt.title("Previsão de temperaturas com SARIMA", fontsize=20)

plt.show()
print('all_data:     ', all_data.index[0], '->', all_data.index[-1])

print('treino:       ', treino.index[0], '->', treino.index[-1])

print('teste:        ', teste.index[0], '->', teste.index[-1])

print('fitted_series:', fitted_series.index[0], '->', fitted_series.index[-1])

print('lower_series: ', lower_series.index[0], '->', lower_series.index[-1])

print('upper_series: ', upper_series.index[0], '->', upper_series.index[-1])
# recuperar última temperatura do treino

ultima_data = treino.index[-1]

ultimo_valor = treino[ultima_data]

primeiro_dia = pd.Series(ultimo_valor, index=[ultima_data])
freq_data = primeiro_dia.append(fitted_series)

print('Intervalo de datas:', freq_data.index[0], '->', freq_data.index[-1])
# obter intervalo de datas necessário

data_inicio = treino.index[-1] #teste.index[0]

data_final = all_data.index[-1] #teste.index[-1]

print('Intervalo necessário:', data_inicio, '=>', data_final)
# interpolar dados para obter valores diários

pred_data = freq_data.resample('D').interpolate(method='cubic')



# restringir ao intervalo de datas esperado

pred_data = pred_data[data_inicio:data_final]



# converter temperaturas novamente para graus Celsius

#pred_data['temperature'] -= 273.15



# exibir informações do dataframe

pred_data = pred_data.to_frame(name='temperature')

pred_data.info()

pred_data.head()
# selecionar dados reais

real_data = all_data[data_inicio:data_final]



# exibir informações do dataframe

real_data.info()

real_data.head()
# plotar gráfico comparando valores reais e previstos

plt.figure(figsize=(16, 9))



real_data['temperature'].plot(alpha=0.5, style='-')

pred_data['temperature'].plot(style=':')

#lower_series.plot(style='--')

#upper_series.plot(style='--')



plt.legend(['real', 'previsto'], loc='upper left')
def rmse(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((predictions - targets) ** 2))



def rmsle(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))
print('RMSE:', rmse(pred_data['temperature'], real_data['temperature']))