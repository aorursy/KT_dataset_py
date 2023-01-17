# importar pacotes necessários

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from fbprophet import Prophet
#prefixo_arquivos = ''

prefixo_arquivos = '../input/serpro-weather/'
# carregar arquivo de dados de treino

train_data = pd.read_csv(prefixo_arquivos + 'weather-train.csv', index_col='date', parse_dates=['date'])

train_data.info()

train_data.head()
# carregar arquivo de dados de teste

test_data = pd.read_csv(prefixo_arquivos + 'weather-test.csv', index_col='date', parse_dates=['date'])

test_data.info()

test_data.head()
# ajustar dados de treino para o formato do Prophet

data2 = train_data[['temperature']]

data2 = data2.reset_index()

data2.columns = ['ds', 'y']

data2.head()
data2.tail()
# criar e treinar o modelo

model = Prophet(daily_seasonality=False)

model.fit(data2)
# criar série com dados futuros (2 anos)

future = model.make_future_dataframe(periods=365*2)

future.tail()
# realizar previsão com dados futuros

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# dividir os dados em 80% + 20%

divisao = int(data2.shape[0] * 4 / 5)

data2a = data2[:divisao]

data2b = data2[divisao:]

print(data2.shape, '=', data2a.shape, '+', data2b.shape)
data2a.info()

data2a.head()
# criar e treinar o modelo

model = Prophet(daily_seasonality=False)

model.fit(data2a)
# preparar dados futuros

future = data2b.drop(['y'], axis=1)

future.info()

future.head()
# realizar a previsão

forecast = model.predict(future)

forecast[['ds', 'yhat']].tail()
# mesclar os dois dataframes novamente

data3 = data2b.merge(forecast)[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]

data3['diff'] = abs(data3['y'] - data3['yhat'])

data3.info()

data3.head()
# plotar gráfico comparando valores reais e previstos

plt.figure(figsize=(16, 9))



data3['y'].plot(alpha=0.5, style='-')

data3['yhat'].plot(style=':')

data3['yhat_lower'].plot(style='--')

data3['yhat_upper'].plot(style='--')



plt.legend(['real', 'previsto', 'pmenor', 'pmaior'], loc='upper left')
def rmse(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((predictions - targets) ** 2))



def rmsle(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))
print('RMSE:', rmse(data3['yhat'], data3['y']))
# criar dados futuros a partir dos dados de teste

future_data = pd.DataFrame(test_data.index.values, columns=['ds'])

future_data.info()

future_data.head()
# realizar a previsão

forecast = model.predict(future_data)

forecast[['ds', 'yhat']].tail()
# construir dataframe com previsão

pred_data = pd.DataFrame({

    'date': forecast['ds'],

    'temperature': forecast['yhat']

})

pred_data.set_index('date', inplace=True)

pred_data.info()

pred_data.head()
# criar diretório de submissões

!dd="submissions/"; if [ ! -d $dd ]; then mkdir $dd; fi
# gravar arquivo CSV com os resultados

nome_arquivo = 'weather-submission-prophet.csv'

#nome_arquivo = 'submissions/weather-submission-prophet.csv'

pred_data.to_csv(nome_arquivo)

print('Arquivo gravado com sucesso:', nome_arquivo)