import os
import sys
import pandas as pd
import numpy as np
from fbprophet import Prophet
daily_data = pd.read_csv('/kaggle/input/daily.csv')
govt_measures = pd.read_excel('/kaggle/input/acaps_covid19_government_measures_dataset.xlsx', sheet_name='Database')
# Convert date to datetime object 
daily_data['date'] = pd.to_datetime(daily_data['date'], format='%Y%m%d')
daily_data.head()
govt_measures.head()
# Filter only for United States
govt_measures = govt_measures[govt_measures['COUNTRY'] == 'United States of America']
daily_data.merge(govt_measures, left_on='date', right_on='DATE_IMPLEMENTED', how='left')
daily_data.rename({'date':'ds','positive':'y'},axis=1,inplace=True)
model = Prophet()
model.fit(daily_data[['ds','y']])
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
