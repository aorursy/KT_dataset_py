import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px

from collections import namedtuple
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

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['Confirmed'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=recovered['Date'],

                y=recovered['Recovered'],

                name='Recovered',

                marker_color='Green'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Bar Chart)',

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

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
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
df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]

df_temp = df.copy()

df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)

df_latlong = pd.merge(df_temp, df_confirmed, on=["Country", "Province/State"])
fig = px.density_mapbox(df_latlong, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Province/State", 

                        hover_data=["Confirmed","Deaths","Recovered"], 

                        animation_frame="Date",

                        color_continuous_scale="Portland",

                        radius=7, 

                        zoom=0,height=700)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
confirmed = df2.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()

deaths = df2.groupby(['Date', 'Country']).sum()[['Deaths']].reset_index()

recovered = df2.groupby(['Date', 'Country']).sum()[['Recovered']].reset_index()
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]

deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]

recovered = recovered[(recovered['Date']==latest_date)][['Country', 'Recovered']]
all_countries = confirmed['Country'].unique()

print("Number of countries/regions with cases: " + str(len(all_countries)))

print("Countries/Regions with cases: ")

for i in all_countries:

    print("    " + str(i))
print(list(country.name for country in pycountry.countries))
print('UK' in list(country.name for country in pycountry.countries))

print('United Kingdom' in list(country.name for country in pycountry.countries))
confirmed2 = confirmed.copy()

deaths2 = deaths.copy()

recovered2 = recovered.copy()

bubble_plot_dfs = [confirmed2, deaths2, recovered2]

for df_ in bubble_plot_dfs:

    df_["Country"].replace({'Mainland China': 'China'}, inplace=True)

    df_["Country"].replace({'UK': 'United Kingdom'}, inplace=True)

    df_["Country"].replace({'US': 'United States'}, inplace=True)
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

confirmed2["iso_alpha"] = confirmed2["Country"].map(countries.get)

deaths2["iso_alpha"] = deaths2["Country"].map(countries.get)

recovered2["iso_alpha"] = recovered2["Country"].map(countries.get)
plot_data_confirmed = confirmed2[["iso_alpha","Confirmed", "Country"]]

plot_data_deaths = deaths2[["iso_alpha","Deaths"]]

plot_data_recovered = recovered2[["iso_alpha","Recovered"]]
fig = px.scatter_geo(plot_data_confirmed, locations="iso_alpha", color="Country",

                     hover_name="iso_alpha", size="Confirmed",

                     projection="natural earth", title = 'Worldwide Confirmed Cases')

fig.show()
fig = px.scatter_geo(plot_data_deaths, locations="iso_alpha", color="Deaths",

                     hover_name="iso_alpha", size="Deaths",

                     projection="natural earth", title="Worldwide Death Cases")

fig.show()
fig = px.scatter_geo(plot_data_recovered, locations="iso_alpha", color="Recovered",

                     hover_name="iso_alpha", size="Recovered",

                     projection="natural earth", title="Worldwide Recovered Cases")

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
forecast_components = m.plot_components(forecast)
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
forecast_components = m.plot_components(forecast)
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
forecast_components = m.plot_components(forecast)
days_to_forecast = 7 # changable

first_forecasted_date = sorted(list(set(df2['Date'].values)))[-days_to_forecast]



print('The first date to perform forecasts for is: ' + str(first_forecasted_date))
confirmed_df = df2[['SNo', 'Date','Province/State', 'Country', 'Confirmed']]

confirmed_df
deaths_df = df2[['SNo', 'Date','Province/State', 'Country', 'Deaths']]

deaths_df
recovered_df = df2[['SNo', 'Date','Province/State', 'Country', 'Recovered']]

recovered_df
def make_forecasts(all_countries, confirmed_df, deaths_df, recovered_df, days_to_forecast, first_forecasted_date, mode):

    forecast_dfs = []

    absolute_errors = [] # collate absolute errors so that we can find MAE later on

    counter = 0 # arbitrary counter to output snippet of result_val_df only once, when it is 1

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



                        state_df_val = state_df[(state_df['ds'] >= pd.to_datetime(first_forecasted_date))] # validation set

                        state_df = state_df[(state_df['ds'] < pd.to_datetime(first_forecasted_date))] # train set



                        if mode == 'default':

                            m = Prophet()

                        elif mode == 'custom':

                            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)

                            m.add_seasonality(name='monthly', period=30.5, fourier_order=10)

                            m.add_seasonality(name='weekly', period=7, fourier_order=21)

                            m.add_seasonality(name='daily', period=1, fourier_order=3)

                        m.fit(state_df)

                        future = m.make_future_dataframe(periods=days_to_forecast)

                        forecast = m.predict(future)



                        # evaluate forecasts with validation set and save absolute errors to absolute_errors

                        forecast_df = forecast[['ds', 'yhat']]

                        result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]

                        result_val_df = result_df.merge(state_df_val, on=['ds'])

                        result_val_df['abs_diff'] = (result_val_df['y'] - result_val_df['yhat']).abs()

                        absolute_errors += list(result_val_df['abs_diff'].values)



                        # the following allows the user to check the output at particular checkpoints. Feel free to change!

                        if counter == 1:

                            print('Printing snippet of result_val_df: \n')

                            print(result_val_df)

                            print(absolute_errors)

                        counter += 1



                        # save results to dataframe

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

    results_tup = namedtuple('results_tup', ['forecast_dfs', 'absolute_errors'])

    return results_tup(forecast_dfs, absolute_errors)
forecast_results = make_forecasts(all_countries, confirmed_df, deaths_df, recovered_df, days_to_forecast, first_forecasted_date, 'default')

absolute_errors = forecast_results.absolute_errors
forecast_results.forecast_dfs[0].tail(days_to_forecast) # example of a forecast
forecasts_final = pd.concat(forecast_results.forecast_dfs, axis=0)

forecasts_final.sort_values(by='ds')

forecasts_final = forecasts_final[['ds', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

forecasts_final.rename(columns={'ds':'ObservationDate'}, inplace=True)

for case_type in ['Confirmed', 'Deaths', 'Recovered']:

    forecasts_final[case_type] = forecasts_final[case_type].round() # round forecasts to integer as humans cannot be floats

    forecasts_final[forecasts_final[case_type] < 0] = 0 # replace negative forecasts to zero



forecasts_final
forecasts_final.to_csv("forecasts_default_prophet.csv", index=False) # save forecasts to CSV
forecast_results_custom = make_forecasts(all_countries, confirmed_df, deaths_df, recovered_df, days_to_forecast, first_forecasted_date, 'custom')

absolute_errors_custom = forecast_results_custom.absolute_errors

forecast_dfs_custom = forecast_results_custom.forecast_dfs
forecasts_final_custom = pd.concat(forecast_dfs_custom, axis=0)

forecasts_final_custom.sort_values(by='ds')

forecasts_final_custom = forecasts_final_custom[['ds', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

forecasts_final_custom.rename(columns={'ds':'ObservationDate'}, inplace=True)

for case_type in ['Confirmed', 'Deaths', 'Recovered']:

    forecasts_final_custom[case_type] = forecasts_final_custom[case_type].round() # round forecasts to integer as humans cannot be floats

    forecasts_final_custom[forecasts_final_custom[case_type] < 0] = 0 # replace negative forecasts to zero



forecasts_final_custom
forecasts_final_custom.to_csv("forecasts_custom_prophet.csv", index=False) # save forecasts to CSV
N = len(absolute_errors)

mean_absolute_error = sum(absolute_errors)/N

print('The mean absolute error for ' + str(days_to_forecast) + ' days of forecasts with the default Prophet model is: ' + str(round(mean_absolute_error, 2))) # round to 2 decimal places
mean_absolute_error_custom = sum(absolute_errors_custom)/N

print('The mean absolute error for ' + str(days_to_forecast) + ' days of forecasts with the custom Prophet model is: ' + str(round(mean_absolute_error_custom, 2))) # round to 2 decimal places