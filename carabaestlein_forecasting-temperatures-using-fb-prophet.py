import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



from datetime import datetime

from fbprophet import Prophet



df = pd.read_csv('../input/GlobalTemperatures.csv')
df['date'] = pd.to_datetime(df['dt'])

df['year'] = df['date'].map(lambda x: x.year)

df['month'] = df['date'].map(lambda x: x.month)
plt.plot(df['year'], df['LandAverageTemperature'])
min_year = df['year'].min()

max_year = df['year'].max()

length = max_year - min_year + 1

years = [min_year]

for i in range(1,length):

    years.append(min_year + i) 
avg_temp = df['LandAverageTemperature'].groupby(df['year']).mean()
plt.scatter(years, avg_temp)
avg_df = pd.DataFrame()

avg_df['y']=avg_temp

avg_df['Year']=years

avg_df['ds']=pd.to_datetime(avg_df['Year'], format='%Y')

length
m = Prophet(n_changepoints=1)

m.fit(avg_df)

future = m.make_future_dataframe(periods=25, freq = 'Y')

forecast = m.predict(future)

m.plot(forecast)
m.plot_components(forecast)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]