import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_confirmed.head()
df.head()
df.tail()
df2 = df.groupby(["Date", "Country", "Province/State"])[['SNo', 'Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df2
df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby('Date').sum()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['Date'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=7)

future_deaths = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=7)

future_recovered = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast)
days_to_forecast = 7 # changable
confirmed_df = df2[['SNo', 'Date','Province/State', 'Country', 'Confirmed']]

confirmed_df
deaths_df = df2[['SNo', 'Date','Province/State', 'Country', 'Deaths']]

deaths_df
recovered_df = df2[['SNo', 'Date','Province/State', 'Country', 'Recovered']]

recovered_df
forecast_dfs = []

for country in all_countries:

    try:

        assert(country in confirmed_df['Country'].values)

        print('Country ' + str(country) + ' is listed! ')

        country_confirmed_df = confirmed_df[(confirmed_df['Country'] == country)]

        country_deaths_df = deaths_df[(deaths_df['Country'] == country)]

        country_recovered_df = recovered_df[(recovered_df['Country'] == country)]

        country_dfs = [('Confirmed', country_confirmed_df), 

                       ('Deaths', country_deaths_df), 

                       ('Recovered', country_recovered_df)]

        states_in_country = country_confirmed_df['Province/State'].unique()

        for state in states_in_country:

            try:

                state_dfs = [] # to store forecasts for Confirmed, Deaths and Recovered

                

                assert(state in country_confirmed_df['Province/State'].values)

                

                # make forecasts for each case type (Confirmed, Deaths, Recovered)

                for country_df_tup in country_dfs:

                    case_type = country_df_tup[0]

                    country_df = country_df_tup[1]

                    state_df = country_df[(country_df['Province/State'] == state)]



                    # data preparation for forecast with Prophet at state level

                    state_df = state_df[['Date', case_type]]

                    state_df.columns = ['ds','y']

                    state_df['ds'] = pd.to_datetime(state_df['ds'])

                    m = Prophet()

                    m.fit(state_df)

                    future = m.make_future_dataframe(periods=days_to_forecast)

                    forecast = m.predict(future)

                    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



                    # save results to dataframe

                    forecast_df = forecast[['ds', 'yhat']]

                    forecast_df['Province/State'] = state

                    forecast_df['Country/Region'] = country

                    forecast_df.rename(columns={'yhat':case_type}, inplace=True)

                    state_dfs += [forecast_df.tail(days_to_forecast)]

                

                merged_df = state_dfs[0].merge(state_dfs[1],on=['ds', 'Province/State', 'Country/Region']).merge(state_dfs[2],on=['ds', 'Province/State', 'Country/Region'])

                forecast_dfs += [merged_df]

            except:

                continue

    except:

        print('Country ' + str(country) + ' is not listed! ')

        continue
forecast_dfs[0].tail(days_to_forecast) # example of a forecast
forecasts_final = pd.concat(forecast_dfs, axis=0)

forecasts_final.sort_values(by='ds')

forecasts_final = forecasts_final[['ds', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

forecasts_final.rename(columns={'ds':'ObservationDate'}, inplace=True)

for case_type in ['Confirmed', 'Deaths', 'Recovered']:

    forecasts_final[case_type] = forecasts_final[case_type].round() # round forecasts to integer as humans cannot be floats

    forecasts_final[forecasts_final[case_type] < 0] = 0 # replace negative forecasts to zero



forecasts_final
forecasts_final.to_csv("forecasts.csv", index=False) # save forecasts to CSV