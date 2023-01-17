import pandas as pd

from fbprophet import Prophet

import os
os.listdir('../input/avocado-prices')
df =  pd.read_csv(r'../input/avocado-prices/avocado.csv', error_bad_lines = False, encoding='latin-1')
df.head()
df = df.drop(['Unnamed: 0','Total Volume','4046','4225', '4770', 'Total Bags','Small Bags','Large Bags','XLarge Bags','type','year','region'], axis=1)
df.head()
df.columns = ['ds','y']
p = Prophet()

p.fit(df)
future = p.make_future_dataframe(periods = 365, include_history = True)

forecast = p.predict(future)
figure = p.plot(forecast, xlabel='Date', ylabel='Average Price ($)')
figure2 = p.plot_components(forecast)