!pip install fbprophet
from fbprophet import Prophet

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

base_test_df = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df.head()
train_df_baseline = df.rename(columns={"datetime": "ds", "count":"y"})

train_df_baseline.drop(columns = ['registered', 'casual'])

test_df_baseline = base_test_df.rename(columns={"datetime": "ds"})
m = Prophet(changepoint_prior_scale=1.5)

m.fit(train_df_baseline)
forecast = m.predict(test_df_baseline)

forecast
fig1 = m.plot(forecast)
def create_submission(forecast):

    result = forecast[['ds', 'trend']]

    result = result.rename(columns = { 'ds': 'datetime', 'trend':'count' })

    result.set_index("datetime", inplace = True) 

    result.to_csv('submission.csv')
create_submission(forecast)
train_df_casual = df.rename(columns={"datetime": "ds", "casual":"y"})

test_df_casual = base_test_df.rename(columns={"datetime": "ds"})
train_df_casual
train_df_casual = train_df_casual.drop(columns = ['registered', 'count'])
train_df_casual
casual_model = Prophet()

casual_model.fit(train_df_casual)

forecast = casual_model.predict(test_df_casual)

forecast
casual_model.plot(forecast)
train_df_registered = df.rename(columns={"datetime": "ds", "registered":"y"})

test_df_registered = base_test_df.rename(columns={"datetime": "ds"})

train_df_registered = train_df_registered.drop(columns = ['casual', 'count'])



registered_model = Prophet()

registered_model.fit(train_df_registered)

forecast_registered = registered_model.predict(test_df_registered)

registered_model.plot(forecast_registered)
forecast
forecast_registered
ultimate_forecast = forecast_registered['trend'] + forecast['trend']
ultimate_forecast
frame = {'count': ultimate_forecast} 

result = pd.DataFrame(frame)
result['datetime'] = base_test_df.datetime

result.set_index("datetime", inplace = True) 

result.to_csv('submission.csv')