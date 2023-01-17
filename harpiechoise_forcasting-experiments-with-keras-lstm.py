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
df = pd.read_csv('/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AMD_data.csv')
df.head()
date = df['date']

open_AMD = df['open']

low_AMD = df['low']

high_AMD = df['high']

close_AMD = df['close']

volume_AMD = df['volume']
import plotly.graph_objs as go

fig = go.Figure(data=go.Candlestick(x=date, open=open_AMD,low=low_AMD, high=high_AMD, close=close_AMD))

fig.update_layout(title='Precio historico AMD', xaxis_title='Fecha', yaxis_title='Varianza entre los precios de Apertura, Altos, Cierre')

fig.show()
from plotly.subplots import make_subplots

fig = make_subplots(rows=5, cols=1, subplot_titles=('Precio apertura Historico AMD', 'Precio cierre Historico AMD', 'Precio alto Historico AMD', 'Precio bajo Historico AMD', 'Volumen historico AMD'))

fig.append_trace(go.Scatter(x=date, y=open_AMD), row=1, col=1)

fig.append_trace(go.Scatter(x=date, y=close_AMD), row=2, col=1)

fig.append_trace(go.Scatter(x=date, y=low_AMD), row=3, col=1)

fig.append_trace(go.Scatter(x=date, y=high_AMD), row=4, col=1)

fig.append_trace(go.Scatter(x=date, y=volume_AMD), row=5, col=1)

fig.update_layout(height=1500)

fig.show()
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

generator = TimeseriesGenerator(close_AMD, close_AMD, length=2, batch_size=1)
for i in range(2):

    x, y = generator[i]

    len(x)

    print(f'{x[0][:3]}... => {y}\nDimensiones de X:{x.shape}\nDimensiones de y: {y.shape} ')
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

generator = TimeseriesGenerator(close_AMD, close_AMD, length=10, batch_size=1)

X = []

y = []

for i in range(len(generator)):

    x_gen, y_gen = generator[i]

    X.append(x_gen)

    y.append(y_gen)
import numpy as np

X = np.array(X)

y = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.models import Sequential

model = Sequential()

model.add(LSTM(128, activation='relu', input_shape=(1, 10)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=600)
predict = model.predict(X)
predict = pd.DataFrame(data={'Dato Real': y.reshape(1249), 'Dato Predecido':predict.reshape(1249)})
fig = go.Figure(data=[go.Line(y=predict[predict.columns[0]], name='Datos Reales (Real Data)'), go.Line(y=predict[predict.columns[1]], name='Dato Predecido (Predicted Data)')])

fig.update_layout(title='Predicciones')

fig.show()
model.evaluate(X_test, y_test)
fig = go.Figure(go.Line(y=history.history['loss'] ))

fig.update_layout(title='Perdida (Model Loss)')

fig.show()
data_NVDA = pd.read_csv('/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/NVDA_data.csv')

X_n = data_NVDA['close'].values

amd_nvda = np.stack((X_n, close_AMD), axis=1)

generator = TimeseriesGenerator(amd_nvda, close_AMD, length=20, batch_size=1)

for i in range(2):

    x, y = generator[i]

    len(x)

    print(f'{x[0][:3]}... => {y}\nDimensiones de X:{x.shape}\nDimensiones de y: {y.shape} ')
X = []

y = []

for i in range(len(generator)):

    x_gen, y_gen = generator[i]

    X.append(x_gen)

    y.append(y_gen)

X = np.array(X)

y = np.array(y)

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train = X_train.reshape(991, 20, 2)

reductor = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=15, verbose=1, min_lr=0.000001)

model = Sequential()

model.add(LSTM(256, activation='relu', input_shape=(20, 2)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=600, callbacks=[reductor])
X_test = X_test.reshape(X_test.shape[0], 20, 2)



model.evaluate(X_test, y_test)

predict = model.predict(X.reshape(X.shape[0], 20, 2))

predict = pd.DataFrame(data={'Dato Real': y.reshape(1239), 'Dato Predecido':predict.reshape(1239)})

fig = go.Figure(data=[go.Line(y=predict[predict.columns[0]], name='Datos Reales (Real Data)'), go.Line(y=predict[predict.columns[1]], name='Dato Predecido (Predicted Data)')])

fig.update_layout(title='Predicciones')

fig.show()
fig = go.Figure(go.Line(y=history.history['loss'] ))

fig.update_layout(title='Perdida (Model Loss)')

fig.show()
generator = TimeseriesGenerator(amd_nvda, close_AMD, length=80, batch_size=1)

X = []

y = []

for i in range(len(generator)):

    x_gen, y_gen = generator[i]

    X.append(x_gen)

    y.append(y_gen)

X = np.array(X)

y = np.array(y)

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv1D, MaxPool1D, TimeDistributed, Flatten 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

reductor = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=15, verbose=1, min_lr=0.000001)

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(1, 80, 2)))

model.add(TimeDistributed(MaxPool1D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=600, callbacks=[reductor])
predict = model.predict(X)

predict = pd.DataFrame(data={'Dato Real': y.reshape(1179), 'Dato Predecido':predict.reshape(1179)})

fig = go.Figure(data=[go.Line(y=predict[predict.columns[0]], name='Datos Reales (Real Data)'), go.Line(y=predict[predict.columns[1]], name='Dato Predecido (Predicted Data)')])

fig.update_layout(title='Predicciones')

fig.show()




from tensorflow.keras.layers import ConvLSTM2D

generator = TimeseriesGenerator(close_AMD, close_AMD, length=80, batch_size=1)

X = []

y = []

for i in range(len(generator)):

    x_gen, y_gen = generator[i]

    X.append(x_gen)

    y.append(y_gen)

X = np.array(X)

y = np.array(y)

X = X.reshape(1179,1,1, 80, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)



model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(1, 1, 80, 1)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=600, callbacks=[reductor])
predict = model.predict(X.reshape(X.shape[0],1,1, 80, 1))

predict = pd.DataFrame(data={'Dato Real': y.reshape(1179), 'Dato Predecido':predict.reshape(1179)})

fig = go.Figure(data=[go.Line(y=predict[predict.columns[0]], name='Datos Reales (Real Data)'), go.Line(y=predict[predict.columns[1]], name='Dato Predecido (Predicted Data)')])

fig.update_layout(title='Predicciones')

fig.show()