import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=3)
bq_assistant.table_schema("crime")
query6 =  """SELECT count(*) as y, DATE(date) as ds
 from
    bigquery-public-data.chicago_crime.crime
 group by ds
 order by ds
  limit 100000
        """
crime_day = chicago_crime.query_to_pandas_safe(query6)

crime_per_day = crime_day[['ds', 'y']]

crime_per_day.tail()

import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
crime_per_day['ds'] = pd.to_datetime(crime_per_day['ds'])
train = crime_per_day.iloc[0:7170] #1000 treningsdager
crime_per_day.plot(x='ds', y='y', label='Antall', xlim=('2001-01-01', '2020-10-10'))

#Resultat uten ekstern variabel
test = crime_per_day.iloc[7170:]
mNew = Prophet(seasonality_mode='multiplicative')
mNew.fit(train)
future = mNew.make_future_dataframe(periods=60, freq='D') #Forecaster 60 dager "fram" i tid
forecast = mNew.predict(future)
ax = forecast.plot(x='ds', y='yhat', label='Hva prophet har forecastet', legend=True)
test.plot(x='ds', y='y', label='Fasit antall', legend=True, ax=ax, xlim=('2020-08-20', '2020-10-10'))
mNew.plot_components(forecast)
#Måle feil slik at vi kan sammenligne modellene
from sklearn.metrics import mean_squared_error

predictions = forecast.iloc[len(train): len(test)+len(train)]['yhat']
error = mean_squared_error(test['y'], predictions)
print(f'prophet MSE Error: {error:11.10}')


#!pip install pmdarima
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
#from pmdarima import auto_arima # for determining ARIMA orders
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
#def arima(csv, start, training_start, training_end, p, q, d):
    
crimes_per_day = crime_per_day.set_index('ds')
crimes_per_day_copy = crimes_per_day.copy()
fig = seasonal_decompose(krim['y'], model='multiplicative').plot()
train = crimes_per_day_copy.iloc[0:7170]
test = crimes_per_day_copy.iloc[7170:]
model = ARIMA(train['y'], order=(1, 1, 1))
result = model.fit()
print(result.summary())

# Obtain predicted values

start = len(train)
end = len(train) + len(test) -1
predictions = result.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')


# Plot predictions against known values
title = 'ARIMA forecasting'
ylabel = 'Antall lovbrudd'
xlabel = ''
ax = test['y'].plot(legend=True, figsize=(12, 6), title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x', tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

# Evaluate the model
error = mean_squared_error(test['y'], predictions)
print(f'ARIMA(1,1,1) MSE Error: {error:11.10}')
#Med ekstern variabel

import pandas as pd
holidays = pd.read_csv('../input/holidaydata/us-holiday-dates-2010-2020-QueryResult.csv')
#holidays = holidays["day_date"]
holidays = pd.to_datetime(holidays["day_date"])

holidays_list = holidays.tolist()


for ind in crime_per_day.index:
    
    if(crime_per_day['ds'][ind] in holidays_list):
        crime_per_day['w'] = 1
    else:
        crime_per_day['w'] = 0
        
crime_per_day.head()

#Mulig bare denne bare gir 0. Da gir ikke prophet nedenfor mye mening
    
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

crime_per_day['ds'] = pd.to_datetime(crime_per_day['ds'])
train = crime_per_day.iloc[:7170]
test = crime_per_day.iloc[7170:]
mNew = Prophet()
mNew.add_regressor('w', prior_scale=0.9, standardize=True, mode='multiplicative')
mNew.add_seasonality(name='weekly', period=7, fourier_order=5)

mNew.fit(train)
future = mNew.make_future_dataframe(periods= 53, freq='D')


future['w'] = crime_per_day['w'].values
forecast = mNew.predict(future)
ax = forecast.plot(x='ds', y='yhat', label='Hva prophet har forecastet', legend=True)
#Skal lage en vertikal strek for hver helligdag.
#for ind in train.index:
#    print(train['w'][ind])
#    ax.axvline(x=x, color='k', alpha = 0.3);
test.plot(x='ds', y='y', label='Fasit', legend=True, ax=ax, xlim=('2020-08-20', '2020-10-10'))
## EVALUATE GOODNESS
predictions = forecast.iloc[len(train): len(test)+len(train)]['yhat']
error = mean_squared_error(test['y'], predictions)
print(f'prophet MSE Error: {error:11.10}')