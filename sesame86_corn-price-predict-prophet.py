import pandas as pd
from fbprophet import Prophet
corn_data = pd.read_csv("../input/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
corn_data
corn_data['date'] = pd.to_datetime(corn_data['date'])
date_price = corn_data[['date', 'price']].reset_index(drop=True)
date_price.plot(x='date', y='price', kind="line")
date_price = date_price.rename(columns={'date':'ds', 'price':'y'})
m = Prophet()
m.fit(date_price)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast.tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)