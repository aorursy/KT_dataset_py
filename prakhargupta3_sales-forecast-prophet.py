import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
 
%matplotlib inline
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('fivethirtyeight')
sales_df = pd.read_csv('../input/train.csv', index_col='date', parse_dates=True)
sales_df = sales_df.loc[sales_df['item'] == 1]
sales_df = sales_df.loc['2014-01-01':'2018-01-01']
sales_df.drop(['store', 'item'], axis=1, inplace=True)
# sales_df = sales_df.sort_index()
sales_df = sales_df.groupby(sales_df.index)[['sales']].sum()
sales_df.head()

sales_df.tail()
df = sales_df.reset_index()
df=df.rename(columns={'date':'ds', 'sales':'y'})
df.head()
df.set_index('ds').y.plot(grid=True)
df['y'] = np.log(df['y'])
df.set_index('ds').y.plot()
model = Prophet(daily_seasonality=True,seasonality_prior_scale=0.5)
model.fit(df);
future = model.make_future_dataframe(periods=60, freq = 'd')
future.tail()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
df.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)
viz_df = sales_df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')

viz_df.head()

viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
viz_df.head()
viz_df[['sales', 'yhat_rescaled']].plot()
sales_df.index = pd.to_datetime(sales_df.index) #make sure our index as a datetime object
connect_date = sales_df.index[-2] #select the 2nd to last date

mask = (forecast.index > connect_date)
predict_df = forecast.loc[mask]

predict_df.head()

viz_df = sales_df.join(predict_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
viz_df['yhat_scaled']=np.exp(viz_df['yhat'])
viz_df.head()

viz_df.tail()

fig, ax1 = plt.subplots()
ax1.plot(viz_df.sales)
ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Sales (Blue) vs Sales Forecast (Black)')
ax1.set_ylabel('Sales')
ax1.set_xlabel('Date')
ax1.set_xlim(['2017-11-01', '2018-03-01'])
#fig.set_xlim(['2017-11-01', '2018-03-01'])

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual Sales') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecasted Sales') #change the legend text for 2nd plot