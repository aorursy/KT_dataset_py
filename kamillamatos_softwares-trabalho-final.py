# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Carregando a base.

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
#Verificando o carregamento da base.

df.shape
# Conferindo amostra dos dados.

df.sample(5).T
#Inspecionando tipos de variáveis do dataset.

df.info()
#importar Datetime para ajustar formato de variaveis

from datetime import datetime
#Ajustando o tipo da variável ObservationDate para DateTime.

df['ObservationDate'] = pd.to_datetime(df['ObservationDate'],infer_datetime_format=True)
df.sample().T
#Conferindo tipo alterado.

df.info()
# Acrescentar nova coluna 'Sick' que representará a quantidade de doentes.
#Calculo de Sick = Confirmed - Deaths - Recovered

df['Sick'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df.sample(40).T
# Selecionar os dados que serão utilizados para a análise.
#Apenas dados sobre o país Brasil.

newdf = df[df['Country/Region'].str.contains('Brazil')]
newdf.head(10)
#Fazendo corte no dataset, separando apenas as colunas que serão utilizadas.

data = newdf[['ObservationDate', 'Sick']]
#Conferidndo tipos de variáveis utilizadas para criar a série temporal.

data.info()
#Setando a coluna ObservationDate como index.

data.set_index('ObservationDate',inplace=True)
#Tranformando o dataset em série temporal.

ts = data['Sick']
#Plotando a série temporal.

plt.plot(ts)
#Definindo funções.
#Plotagens de Rolling Statistics e resultados dos testes de Dickey-Fulle.

from statsmodels.tsa.stattools import adfuller
def test_stationarity(ts):
    
    #Determing rolling statistics
    rolmean = pd.Series(ts).rolling(window=7).mean()
    rolstd = pd.Series(ts).rolling(window=7).std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Desvio Padrão')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Resultados do Teste de Dickey-Fuller:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estatística de teste','p-valor','Defasagem usada','Número de observações usadas'])
    for key,value in dftest[4].items():
        dfoutput['Valor crítico (%s)'%key] = value
    print (dfoutput)
#Teste de estacionariedade.

test_stationarity(ts)
#Realizando transformação em logaritmo.

ts_log = np.log(ts)
plt.plot(ts_log)
#Dropando valores Na

ts_log.dropna()
#Dropando valores "não válidos".

ts_log.drop(ts_log.index[[0]],inplace=True)

#determinando estatísticas de rolagem.

moving_avg = ts_log.rolling(window=7).mean()  
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
#Mostrando valores que não possuem média.

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(7)
#Usando valores NaN para testar a estacionaridade.

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
#Atribuindo pesos utilizando a média ponderada exponencial.

expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
#Tentando obter estacionariedade.

ts_log_ewma_diff = ts_log - expwighted_avg 
test_stationarity(ts_log_ewma_diff)
#Testando método da diferenciação de primeira ordem para obter estacionariedade.

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
# Verificando a diferenciação por meio de gráficos.

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
#Modelando a tendência e sazonalidade pela método de decomposição.

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log) 
trend = decomposition.trend 
seasonal = decomposition.seasonal 
residual = decomposition.resid 
plt.subplot(411) 
plt.plot(ts_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.tight_layout()
#Verificando a estacionaridade dos resíduos.

ts_log_decompose = residual 
ts_log_decompose.dropna(inplace=True) 
test_stationarity(ts_log_decompose)
#Plotando ACF e PACF.

from statsmodels.tsa.stattools import acf, pacf

#Função de autocorrelação (ACF): É uma medida da correlação entre o TS com uma versão desfasada de si mesmo.

lag_acf = acf(ts_log_diff, nlags=10)

#Função de autocorrelação parcial (FACP): Esta mede a correlação entre o TS com uma versão desfasada de si mesmo, 
#mas depois de eliminar as variações já explicadas pelas comparações intervenientes.

lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')
#Plotando ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plotando PACF: 
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray') 
plt.title('Partial Autocorrelation Function') 
plt.tight_layout()
#Importando modelo ARIMA

from statsmodels.tsa.arima_model import ARIMA
#Utilizando modelo AR.

model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
#Utilizando modelo MA.

model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
##Utilizando modelo combinado ARIMA.

model = ARIMA(ts_log, order=(2, 1, 2)) 
results_ARIMA = model.fit(disp=-1) 
plt.plot(ts_log_diff) 
plt.plot(results_ARIMA.fittedvalues, color='red') 
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#Armazenando resultados previstos

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True) 
print (predictions_ARIMA_diff.head())
#Convertendo a diferenciação de escala logarítmica utilizando a soma cumulativa.

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum() 
print (predictions_ARIMA_diff_cumsum.head())
#Criando uma série com os números base e adicionando as diferenças a eles

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print (predictions_ARIMA_log.head())
#Tomando o expoente e comparando com a série original.

predictions_ARIMA = np.exp(predictions_ARIMA_log) 
plt.plot(ts) 
plt.plot(predictions_ARIMA) 
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))