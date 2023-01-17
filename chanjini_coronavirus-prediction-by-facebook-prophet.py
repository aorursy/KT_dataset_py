from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

import datetime

%matplotlib inline 
import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

%matplotlib inline 
confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

recover = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
# Check the latest dataset's date

confirmed.columns[-1:]
# Make Korea's confirmed cases timeseries dataframe

df_korea = confirmed[confirmed['Country/Region'] == 'Korea, South']

df_korea
confirmed['Country/Region'].unique()
# United State and China's cases are divided into States or Provinces(省). 

# United State's Cases

df_US = confirmed[confirmed['Country/Region'] == 'US']

# df_US



# Following is example of dataframe for Los Angeles

df_LA = df_US[df_US["Province/State"] == "Los Angeles, CA"]

#df_LA



# Rest of the regions are aggregation of nation's entire cases

# Following is example of dataframe of Italy's confirmed Cases

df_IT = confirmed[confirmed['Country/Region'] == 'Italy']

#df_IT
df_korea = df_korea.T[4:]



# Make Korean confirmed timeseries dataframe into two rows: date and number of confirmed cases

df_korea = df_korea.reset_index().rename(columns={'index': 'date', 158: 'confirmed'})



df_korea['date'] = pd.to_datetime(df_korea['date'])



# Check the most recent 5 days of Korean confirmed cases 

df_korea.tail()
# Plot Korean COVID19 confirmed cases 

plt.figure(facecolor='white', figsize=(20, 10))



plt.plot(df_korea.date, df_korea.confirmed, 

            label="confirmed in Korea")

plt.xlabel("Date")

plt.ylabel("Confirmed")

plt.legend()

plt.show()
# Make dataframe for Facebook Prophet prediction model

df_prophet = df_korea.rename(columns={

    'date': 'ds',

    'confirmed': 'y'

})



df_prophet.tail()
m = Prophet(

    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

    changepoint_range=0.98, # place potential changepoints in the first 98% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)

m.fit(df_prophet)
def predict_sequence_full(m, df_prophet, n):

    for i in range(n):

        

        future = m.make_future_dataframe(periods=1)

        forecast = m.predict(future)

        temp = pd.concat([future[-1:], forecast[-1:]['yhat']], axis=1)

        temp = temp.rename(columns = {'yhat':'y'})

        df_prophet = df_prophet.append(temp)

        

        m = Prophet(

            changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

            changepoint_range=0.98, # place potential changepoints in the first 98% of the time series

            yearly_seasonality=False,

            weekly_seasonality=False,

            daily_seasonality=True,

            seasonality_mode='additive'

        )

        m.fit(df_prophet)

        

    return forecast
forecast = predict_sequence_full(m, df_prophet, 7)
forecast.tail(7)
forecast['yhat'].tail(7)
fig = m.plot(forecast)
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)