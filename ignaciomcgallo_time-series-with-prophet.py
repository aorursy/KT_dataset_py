import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator



import os

data = pd.read_csv('../input/emt-dataset/datos_pasajeros.csv',sep=';')

data.head()
data.shape
data.describe()
data.isnull().sum()
type(data.fecha[0])
data['fecha'] = pd.to_datetime(data['fecha'])
type(data.fecha[0])
data.head()
data = data.rename(columns={"fecha":"date","mes":"month","festivo":"holiday","npasajeros":"passengers","intensidad_evento":"event","inten_lluvia":"rain","ocupacion_trafico":"traffic","semana_mes":"week"})
data.head()
print("Starting date of time series: ", data.date.min())

print("Final date of time series:    ", data.date.max())
dates = data.date.values

passengers_n = data.passengers.values
plt.figure(figsize=(15,5))

plt.plot(dates, passengers_n)

plt.title('Passengers average',

          fontsize=20);
plt.figure(figsize=(15,5))

plt.plot(dates, passengers_n, 'o-')

plt.title('Passengers average', fontsize=20)

plt.axis([dates[-150],dates[-1],0,5000]);
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.hist(passengers_n, bins=30)

plt.xlabel('Passengers', fontsize=20)

plt.subplot(1,2,2)

aux = np.log( passengers_n[1:] / passengers_n[0:-1]  )

plt.hist(aux, bins=30)

plt.xlabel('Logarithmic passengers increment', fontsize=20)

plt.show()

print("Passengers average                      :", passengers_n.mean())

print("Logarithmic passanger increment average:", aux.mean())
plt.figure(figsize=(15,5))

plt.plot(dates[1:], aux)

plt.title('Passengers (Logarithmic)',

          fontsize=20);
from fbprophet import Prophet

import logging



logging.getLogger().setLevel(logging.ERROR)
data.head()
aux = data.loc[:,["date","passengers"]]

aux.head()
aux.shape
aux.columns = ['ds', 'y']

aux.head()
aux.shape
df_train = aux[:-30]

df_train.shape
model = Prophet()

model.fit(df_train)
future = model.make_future_dataframe(periods=60)



forecast = model.predict(future)
forecast.tail()
model.plot(forecast)

plt.axvline(x=data.date.values[-30], c="r", ls="--")
model.plot_components(forecast)
def make_comparison_dataframe(historical, forecast):

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(aux, forecast)

cmp_df.head()
def calculate_forecast_errors(df, prediction_size):

    

    df = df.copy()

    

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    predicted_part = df[-prediction_size:]

    

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
for err_name, err_value in calculate_forecast_errors(cmp_df,60).items():

    print(err_name, err_value)
plt.figure(figsize=(17, 8))

plt.plot(cmp_df['yhat'])

plt.plot(cmp_df['yhat_lower'])

plt.plot(cmp_df['yhat_upper'])

plt.plot(cmp_df['y'])

plt.axvline(x=data.date.values[-30], c="r", ls="--")

plt.xlabel('Time')

plt.ylabel('Daily Passanger')

plt.grid(False)

plt.show()