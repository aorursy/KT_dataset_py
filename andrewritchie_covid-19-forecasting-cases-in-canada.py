# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recovered_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')



df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



confirmed_df.rename(columns={'Country/Region':'Country'}, inplace=True)

deaths_df.rename(columns={'Country/Region':'Country'}, inplace=True)

recovered_df.rename(columns={'Country/Region':'Country'}, inplace=True)
df.tail()
df2 = df.groupby(["Date", "Country", "Province/State"])[['SNo', 'Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df2["Active"] = df2['Confirmed']-df2['Deaths']-df2['Recovered']

df["Active"] = df['Confirmed']-df['Deaths']-df['Recovered']

df2
df.groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered','Active']].sum().reset_index()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered','Active']].sum().reset_index()
df.groupby('Date').sum()
confirmed = df2.groupby(['Date', 'Country','Province/State']).sum()[['Confirmed']].reset_index()

deaths = df2.groupby(['Date', 'Country','Province/State']).sum()[['Deaths']].reset_index()

recovered = df2.groupby(['Date', 'Country','Province/State']).sum()[['Recovered']].reset_index()

active = df2.groupby(['Date', 'Country','Province/State']).sum()[['Active']].reset_index()
confirmed
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]

deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]

recovered = recovered[(recovered['Date']==latest_date)][['Country', 'Recovered']]

active = active[(active['Date']==latest_date)][['Country', 'Active']]
# add .query('Country=="Canada"') to create new dataframes containing only Canadian data. Can be changed to other countries if desired.



confirmed = df.query('Country=="Canada"').groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.query('Country=="Canada"').groupby('Date').sum()['Deaths'].reset_index()

recovered = df.query('Country=="Canada"').groupby('Date').sum()['Recovered'].reset_index()

active = df.query('Country=="Canada"').groupby('Date').sum()['Active'].reset_index()      
confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(mcmc_samples = 100, seasonality_mode = 'additive', changepoint_prior_scale=1, interval_width=0.95)

m.fit(confirmed, control={'max_treedepth': 20})

future = m.make_future_dataframe(periods=7)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast, xlabel = 'date', ylabel = 'COVID-19 Confirmed Cases in Canada')
fig1 = m.plot_components(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(mcmc_samples = 100, seasonality_mode = 'additive', changepoint_prior_scale=1,interval_width=0.95)

m.fit(recovered, control={'max_treedepth': 20})

future = m.make_future_dataframe(periods=7)

future_recovered = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast, xlabel = 'Date', ylabel = 'COVID-19 Recovered Cases in Canada')
fig2 = m.plot_components(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(mcmc_samples = 100, seasonality_mode = 'additive', changepoint_prior_scale=1, interval_width=0.95)

m.fit(deaths, control={'max_treedepth': 20})

future = m.make_future_dataframe(periods=7)

future_deaths = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast, xlabel='Date', ylabel='COVID-19 Deaths in Canada')
fig3 = m.plot_components(forecast)
active.columns = ['ds','y']

active['ds'] = pd.to_datetime(active['ds'])

active['floor'] = 0
m = Prophet(mcmc_samples = 100, seasonality_mode = 'additive', changepoint_prior_scale=1, interval_width=0.99)

m.fit(active, control={'max_treedepth': 100})

future = m.make_future_dataframe(periods=30)

future_active = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
active_forecast_plot = m.plot(forecast, xlabel = 'Date', ylabel = 'COVID-19 Active Cases in Canada')
fig4 = m.plot_components(forecast)