# Load standard libraries

import numpy as np

RANDOM_SEED = 1337

np.random.seed(RANDOM_SEED)

import pandas as pd

pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 70)



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()



import gc



print('Loaded libraries')
# Get store id's that will be used for submission. No need for visit_date.

df_sub = pd.read_csv('../input/sample_submission.csv')

df_sub['store_id'] = df_sub['id'].apply(lambda x:x[:-11])

# df_sub['visit_date'] = df_sub['id'].apply(lambda x:pd.to_datetime(x[-10:]))

df_sub = df_sub.set_index('id')

df_sub
# load train data

df_air_visit_data = pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date'])

df_air_visit_data
# Prepare holiday data

# https://facebook.github.io/prophet/docs/seasonality_and_holiday_effects.html

df_holidays = pd.read_csv('../input/date_info.csv', 

                          usecols=['calendar_date', 'holiday_flg'],

                          parse_dates=['calendar_date'])

df_holidays.columns = ['ds', 'holiday']

df_holidays = df_holidays[df_holidays['holiday'] == 1]

df_holidays['holiday'] = 'holiday'

df_holidays
# Prediction

from fbprophet import Prophet

# This is used for suppressing prophet info messages.

import logging

logging.getLogger('fbprophet.forecaster').propagate = False



number_of_stores = df_sub['store_id'].nunique()

date_range = pd.date_range(start=pd.to_datetime('2016-07-01'),

                           end=pd.to_datetime('2017-04-22'))

forecast_days = (pd.to_datetime('2017-05-31')-pd.to_datetime('2017-04-22')).days



for cnt, store_id in enumerate(df_sub['store_id'].unique()):

    print('Predicting %d of %d.'%(cnt, number_of_stores), end='\r')

    data = df_air_visit_data[df_air_visit_data['air_store_id'] == store_id]

    data = data[['visit_date', 'visitors']].set_index('visit_date')

    # Ensure we have full range of dates.

    data = data.reindex(date_range).fillna(0).reset_index()

    data.columns = ['ds', 'y']

    

    m = Prophet(holidays=df_holidays)

    m.fit(data)

    future = m.make_future_dataframe(forecast_days)

    forecast = m.predict(future)

    forecast = forecast[['ds', 'yhat']]

    forecast.columns = ['id', 'visitors']

    forecast['id'] = forecast['id'].apply(lambda x:'%s_%s'%(store_id, x.strftime('%Y-%m-%d')))

    forecast = forecast.set_index('id')

    df_sub.update(forecast)

print('\n\nDone.')
# Make submission

df_sub = df_sub.reset_index()[['id','visitors']]

df_sub['visitors'] = df_sub['visitors'].clip(lower=0)

df_sub.to_csv('submission.csv', index=False)
print('Everything is ok?')