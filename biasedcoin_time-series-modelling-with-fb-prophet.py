%matplotlib inline

import pandas as pd

from fbprophet import Prophet



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
data = pd.read_csv('../input/AirPassengers.csv')
data.head()
data.dtypes
#We see that the Month data is identified as string

#We'll convert it into a DateTime type



data['Month'] = pd.DatetimeIndex(data['Month'])

data.dtypes
data.head()
# Prophet requires that the columns be named as 'ds' and 'y'

data = data.rename(columns={'Month': 'ds',

                        '#Passengers': 'y'})
plt.plot(data['ds'],data['y'])
model = Prophet()
model.fit(data)
future_dates = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future_dates)
p=model.plot_components(forecast)
p=model.plot(forecast)
m = Prophet(seasonality_mode='multiplicative')

m.fit(data)

forecast2 = m.predict(future_dates)

p=m.plot(forecast2)
p=m.plot_components(forecast)