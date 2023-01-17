import numpy as np

import pandas as pd

import pandas_datareader as pdr

from pandas import Timestamp

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns 

from IPython import display

import matplotlib.ticker as mticker

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

try:

  from stockai import Stock

except:

  print('instalando stockai....')

  !pip install --upgrade  stockai

  from stockai import Stock

display.clear_output(wait=True)

plt.style.use(['seaborn-deep'])

mpl.style.use(['seaborn-deep'])

display.clear_output()
#@title Parâmetros de pesquisa. 

#@markdown ---

#@markdown ### Exemplo de ticker: PETR4.SA

ticker = 'PETR4.SA' # @param {type:"string"}

#@markdown ---

start_date = '2008-01-01'  #@param {type:"date"}

end_date = '2019-12-31'  #@param {type:"date"}
stock = Stock(ticker)

prices_list = stock.get_historical_prices(start_date, end_date)

prices_dataframe = pd.DataFrame.from_dict(prices_list)

history_data = prices_dataframe.copy()

history_data.describe()
#Verificando dados não preenchidos..

history_data.isna().any()
#Removendo dados não preenchidos... (excluindo ruidos)

history_data.dropna(inplace=True)

history_data.isna().any()
history_data.info()
history_data = history_data.sort_values('date')

history_data.head(5)
data = history_data.copy()

data['date'] = pd.to_datetime( data['date'], format='%Y/%m/%d')

data.head(1)
plt.clf()

plt.figure(figsize=(15,2),frameon=False)

plt.title('AdjClose')

plt.xlabel('Days', fontsize=14)

plt.ylabel('AdjClose R$',fontsize=14)

data.adjclose.plot(color='C8')

plt.grid(True)

plt.show()
#Cacl o MACD e sinal

#short - Exponetial Moving Avage - EMA 

#Fibonacci 0,1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,

short_EMA = data.adjclose.ewm(span=8,adjust=False).mean()

#long - Exponetial Moving Avage - EMA

long_EMA = data.adjclose.ewm(span=21,adjust=False).mean()

#MACD line

macd = short_EMA - long_EMA

#MACD signal

signal = macd.ewm(span=5,adjust=False).mean()
#create new col

data['macd'] = macd

data['macd_signal'] = signal

# buy and sell an asset

def buy_or_sell(rows):

  buy =[]

  sell =[]

  flag = -1

  for i in range(0, rows.shape[0]):

    if rows[i][0] > rows[i][1]:

      sell.append(np.nan)

      if flag != 1:

        buy.append(rows[i][2])

        flag=1

      else:

         buy.append(np.nan)

    elif rows[i][0] < rows[i][1]:

      buy.append(np.nan)

      if flag != 0:

        sell.append(rows[i][2])

        flag=0

      else:

         sell.append(np.nan)

    else:

       sell.append(np.nan)

       buy.append(np.nan)

  return buy,sell
d = data[(data.shape[0]-233):].copy()

plt.clf()

plt.figure(figsize=(15,4))

plt.plot(d.index, d.macd, label=ticker+' MACD',color='red')

plt.plot(d.index, d.macd_signal, label=ticker+' Signal',color='blue')

plt.legend(loc='upper center')

plt.xlabel('Days', fontsize=14)

plt.grid(True)

plt.show()
data.dropna(inplace=True)
#create buy 

cols=['macd','macd_signal','adjclose']

rows = data[(data.shape[0]-233):][cols].copy()

buy,sell = buy_or_sell(rows.values)

rows['buy_signal'] = buy

rows['sell_signal'] = sell

data.drop("close", axis = 1, inplace = True)
#Visually  stock buy and sell signals

plt.clf()

plt.figure(figsize=(15,5))

plt.plot(rows.index, rows.buy_signal, label='Buy',color='green', marker='^', alpha=2)

plt.plot(rows.index, rows.sell_signal, label='Sell',color='red', marker='*', alpha=2)

plt.plot(rows.index, rows.adjclose, label=ticker+' Close', alpha=0.7)

plt.legend(loc='upper center')

plt.ylabel('Close R$')

plt.xticks(rotation=45)

plt.xlabel('Days')

plt.title('Close Buy & Sell Signals')

plt.grid(True)

plt.show()

#https://www.youtube.com/watch?v=kz_NJERCgm8
#@title ATR - intervalo verdadeiro medio

#@markdown É um indicador de análise técnica que mede a volatilidade do mercado decompondo todo o intervalo de um preço de ativo para esse período. 

#@markdown Especificamente, o ATR é uma medida de volatilidade.

#http://kaushik316-blog.logdown.com/posts/1964522

data['tr1'] = data.high - data.low

data['tr2'] = abs(data.high - data.adjclose.shift(1))

data['tr3'] = abs(data.low - data.adjclose.shift(1))

data['true_range'] = data[['tr1','tr2','tr3']].max(axis=1)

data["tr_avg"] = data.true_range.rolling(min_periods=13, window=13, center=False).mean()

data['atr'] = data["tr_avg"] .ewm(alpha=1/13, adjust=False).mean()

data.drop('tr1',axis=1,inplace=True)

data.drop('tr2',axis=1,inplace=True)

data.drop('tr3',axis=1,inplace=True)

data.drop('tr_avg',axis=1,inplace=True)

data.drop('true_range',axis=1,inplace=True)

data.atr.fillna(0.001,inplace=True)

windows =[5, 8, 13, 21]

for w in windows:

  data['ma_{}days'.format(w)] = data.adjclose.rolling(window=w).mean()

  data['close_{}days'.format(w)] = data.adjclose.shift(w*-1) 

  data['close_{}days_pct'.format(w)] = data.adjclose.pct_change(w)

data.dropna(axis = 1, inplace = True)

data.isna().any()
total=data.isnull().sum().sort_values(ascending=False)

percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing=pd.concat([total,percent], axis=1,keys=['Total','%'])

missing[(missing['%']>0)].head(30)
data_corr = data.copy() 

data_corr = data_corr.reset_index(drop=True)

plt.clf()

plt.figure(figsize=(15,5))

corr =data_corr.corr(method='pearson')

corr = corr[corr>=.2]

sns.heatmap(corr,annot=True,cmap='YlGnBu',fmt='.1f',linewidths=1)
best_corr = corr['adjclose'].sort_values(ascending=False).to_dict()

columns =[]

for key,value in best_corr.items():

  if value > 0 :

    columns.append(key)

    print(key,value)
dataset = data.copy()

dataset.dropna(inplace=True)

train_len =  int(dataset.shape[0] * 0.8) 

train_base = dataset[columns].values

x_train = train_base

y_train = dataset[['adjclose']].values.reshape(-1)



x_test = x_train[train_len:]

y_test = y_train[train_len:]



x_train = x_train[:train_len]

y_train = y_train[:train_len]



# reshape os dados pois o modelo LSTM  espera receber entradas em 3D

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))



x_train.shape, x_test.shape
#Modelo LSTM

import tensorflow as tf

from tensorflow import keras

model = keras.Sequential()



model.add(keras.layers.LSTM(144, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.LSTM(89, return_sequences=True))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.LSTM(55))

model.add(keras.layers.Dense(1))

batch_size = 32

train_size = x_train.shape[0]

model.compile(optimizer='RMSprop', loss='mean_absolute_error', metrics=['mean_absolute_error'])

history = model.fit(x_train, y_train,  verbose=2, epochs=25, batch_size=32)

display.clear_output()
#@title MAE - Erro Absoluto Médio

#@markdown O mae mede a magnitude média dos erros em um conjunto de previsões, sem considerar sua direção.

#@markdown É a média sobre a amostra de teste das diferenças absolutas entre previsão e observação real, onde todas as diferenças individuais têm peso igual.

epochs = list(range(1,(np.array(history.history['mean_absolute_error']).shape[0])+1))

plt.clf()

fig = plt.figure(facecolor='white',figsize=(10,3))

plt.xlabel('Epochs',fontsize=15)

plt.ylabel('MAE')

plt.plot(epochs,history.history['mean_absolute_error'],color='red')

plt.legend(('MAE','MAE'),fontsize=18)

plt.grid(True)

plt.show()
predict = model.predict(x_test)

previsoes = predict.reshape(-1)

# obter o mean absolute error

mae = np.mean(np.abs(previsoes - y_test))

mae
# plot the data

train = dataset[:train_len]

actual = dataset[train_len:]

actual['predictions'] = previsoes# pd.Series(np.random.randn(sLength), index=df1.index)

actual.head(2)
plt.figure(figsize=(15,15))

plt.title("LSTM - Predict Stock")

plt.xlabel('Days', fontsize =18)

plt.ylabel('Close', fontsize=18)

plt.grid(True)

plt.plot(train['adjclose'][2000:])

plt.plot(actual[['adjclose','predictions']])

plt.legend(['Train', 'Actual', 'Prediction'], loc='upper center')

plt.axis([2000,3000,7,31])

plt.annotate("NADA DURA PARA SEMPRE...  ;'-('", xy=(2700, 21), xytext=(2700,14), arrowprops=dict(facecolor='black', width=1.5, shrink=0.1, headwidth=15))

plt.show()