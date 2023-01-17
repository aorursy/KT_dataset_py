! pip install calmap
# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns



from plotnine import *

import calmap



import plotly.express as px

import folium

import plotly.graph_objs as go

from fbprophet import Prophet



# color pallettes

rbg = ['#ff0000','#000000', '#00ff00']

bg = ['#000000', '#00ff00']

grb = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

yrb = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue
novel_corona_cleaned_latest = pd.read_csv("../input/corona-virus-dataset-2019-covid19-latest/novel_corona_cleaned_latest.csv")

time_series_confirmed = pd.read_csv("../input/corona-virus-dataset-2019-covid19-latest/time_series_confirmed.csv")

time_series_deaths = pd.read_csv("../input/corona-virus-dataset-2019-covid19-latest/time_series_deaths.csv")

time_series_recovered = pd.read_csv("../input/corona-virus-dataset-2019-covid19-latest/time_series_recovered.csv")
# importing datasets

full_table = pd.read_csv('../input/corona-virus-dataset-2019-covid19-latest/novel_corona_cleaned_latest.csv', 

                         parse_dates=['Last Update'])

full_table.head()
# checking for missing value

# full_table.isna().sum()
# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values with NA

# full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')
dpc_ship = full_table[full_table['Province/State']=='Diamond Princess cruise ship']

full_table = full_table[full_table['Province/State']!='Diamond Princess cruise ship']

china = full_table[full_table['Country/Region']=='China']

germany = full_table[full_table['Country/Region']=='Germany']

egypt = full_table[full_table['Country/Region']=='Egypt']

other_countries = full_table[full_table['Country/Region']!='China']



full_latest = full_table[full_table['Last Update'] == max(full_table['Last Update'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

germany_latest = full_latest[full_latest['Country/Region']=='Germany']

egypt_latest = full_latest[full_latest['Country/Region']=='Egypt']

other_countries_latest = full_latest[full_latest['Country/Region']!='China']



full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

germany_latest_grouped = germany_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

egypt_latest_grouped = egypt_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

other_countries_latest_grouped = other_countries_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
# All countries with sorted by Confirmed cases

temp_full_confirmed = full_latest_grouped[['Country/Region', 'Confirmed']]

temp_full_confirmed = temp_full_confirmed.sort_values(by='Confirmed', ascending=False)

temp_full_confirmed = temp_full_confirmed.reset_index(drop=True)

temp_full_confirmed.style.background_gradient(cmap='Oranges')
# All countries with sorted by Deaths cases

temp_full_deaths = full_latest_grouped[['Country/Region', 'Deaths']]

temp_full_deaths = temp_full_deaths.sort_values(by='Deaths', ascending=False)

temp_full_deaths = temp_full_deaths.reset_index(drop=True)

temp_full_deaths = temp_full_deaths[temp_full_deaths['Deaths']>0]

temp_full_deaths.style.background_gradient(cmap='Greys')
# All countries with sorted by Recovered cases

temp_full_recovered = full_latest_grouped[['Country/Region','Recovered']]

temp_full_recovered = temp_full_recovered.sort_values(by='Recovered', ascending=False)

temp_full_recovered = temp_full_recovered.reset_index(drop=True)

temp_full_recovered = temp_full_recovered[temp_full_recovered['Recovered']>0]

temp_full_recovered.style.background_gradient(cmap='Greens')
# All countries with sorted by Confirmed cases outside china

temp_full_confirmed_outside_china = other_countries_latest_grouped[['Country/Region', 'Confirmed']]

temp_full_confirmed_outside_china = temp_full_confirmed_outside_china.sort_values(by='Confirmed', ascending=False)

temp_full_confirmed_outside_china = temp_full_confirmed_outside_china.reset_index(drop=True)

temp_full_confirmed_outside_china.style.background_gradient(cmap='Oranges')
# All countries with sorted by Deaths cases outside china

temp_full_deaths_outside_china = other_countries_latest_grouped[['Country/Region', 'Deaths']]

temp_full_deaths_outside_china = temp_full_deaths_outside_china.sort_values(by='Deaths', ascending=False)

temp_full_deaths_outside_china = temp_full_deaths_outside_china.reset_index(drop=True)

temp_full_deaths_outside_china = temp_full_deaths_outside_china[temp_full_deaths_outside_china['Deaths']>0]

temp_full_deaths_outside_china.style.background_gradient(cmap='Greys')
# All countries with sorted by Recovered cases outside china

temp_full_recovered_outside_china = other_countries_latest_grouped[['Country/Region','Recovered']]

temp_full_recovered_outside_china = temp_full_recovered_outside_china.sort_values(by='Recovered', ascending=False)

temp_full_recovered_outside_china = temp_full_recovered_outside_china.reset_index(drop=True)

temp_full_recovered_outside_china.style.background_gradient(cmap='Greens')
temp_all_recovered = full_latest_grouped[full_latest_grouped['Confirmed'] == full_latest_grouped['Recovered']]

temp_all_recovered = temp_all_recovered[['Country/Region', 'Confirmed', 'Recovered']]

temp_all_recovered = temp_all_recovered.sort_values('Confirmed', ascending=False)

temp_all_recovered = temp_all_recovered.reset_index(drop=True)

temp_all_recovered.style.background_gradient(cmap='Greens')
temp_cleared = full_latest_grouped[full_latest_grouped['Confirmed']== full_latest_grouped['Deaths']+ full_latest_grouped['Recovered']]

temp_cleared = temp_cleared[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

temp_cleared = temp_cleared.sort_values('Confirmed', ascending=False)

temp_cleared = temp_cleared.reset_index(drop=True)

temp_cleared.style.background_gradient(cmap='Greens')
temp_daily_all = full_table.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp_daily_all = temp_daily_all.reset_index()

temp_daily_all = temp_daily_all.melt(id_vars="Last Update", value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp_daily_all, x="Last Update", y="value", color='variable', title='Daily Status - All Countries',

             color_discrete_sequence=grb)

fig.update_layout(barmode='group')

fig.show()
temp_daily_outside_china = other_countries.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp_daily_outside_china = temp_daily_outside_china.reset_index()

temp_daily_outside_china = temp_daily_outside_china.melt(id_vars="Last Update", value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp_daily_outside_china, x="Last Update", y="value", color='variable', title='Daily Status - outside China',

             color_discrete_sequence=grb)

fig.update_layout(barmode='group')

fig.show()
temp_daily_germany = germany.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp_daily_germany = temp_daily_germany.reset_index()

temp_daily_germany = temp_daily_germany.melt(id_vars="Last Update", value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp_daily_germany, x="Last Update", y="value", color='variable', title='Daily Status - Germany',

             color_discrete_sequence=grb)

fig.update_layout(barmode='group')

fig.show()
temp_daily_egypt = egypt.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp_daily_egypt = temp_daily_egypt.reset_index()

temp_daily_egypt = temp_daily_egypt.melt(id_vars="Last Update", value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp_daily_egypt, x="Last Update", y="value", color='variable', title='Daily Status - Egypt',

             color_discrete_sequence=grb)

fig.update_layout(barmode='group')

fig.show()
temp_confirmed_by_time_all = full_table.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp_confirmed_by_time_all.head()

temp_confirmed_by_time_all = temp_confirmed_by_time_all.melt(id_vars='Last Update', 

                                                                        value_vars=['Confirmed', 'Deaths', 'Recovered'],

                                                             var_name='Confirmed', value_name='Value')

fig = px.line(temp_confirmed_by_time_all, x="Last Update", y="Value", color='Confirmed', 

              title='Status by time - All Countries',color_discrete_sequence=rbg)

fig.show()
temp_confirmed_by_time_germany = germany.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp_confirmed_by_time_germany.head()

temp_confirmed_by_time_germany = temp_confirmed_by_time_germany.melt(id_vars='Last Update', 

                                                                        value_vars=['Confirmed', 'Deaths', 'Recovered'],

                                                             var_name='Confirmed', value_name='Value')

fig = px.line(temp_confirmed_by_time_germany, x="Last Update", y="Value", color='Confirmed', 

              title='Status by time - Germany',color_discrete_sequence=rbg)

fig.show()
temp_confirmed_by_time_egypt = egypt.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp_confirmed_by_time_egypt.head()

temp_confirmed_by_time_egypt = temp_confirmed_by_time_egypt.melt(id_vars='Last Update', 

                                                                        value_vars=['Confirmed', 'Deaths', 'Recovered'],

                                                             var_name='Confirmed', value_name='Value')

fig = px.line(temp_confirmed_by_time_egypt, x="Last Update", y="Value", color='Confirmed', 

              title='Status by time - Egypt',color_discrete_sequence=rbg)

fig.show()
temp_infection_recovery_by_time_all = full_table.groupby('Last Update').sum().reset_index()

temp_infection_recovery_by_time_all.head()



# adding two more columns

temp_infection_recovery_by_time_all['No. of Deaths to 100 Confirmed Cases'] = round(temp_infection_recovery_by_time_all['Deaths'] / temp_infection_recovery_by_time_all['Confirmed'], 3) * 100

temp_infection_recovery_by_time_all['No. of Recovered to 100 Confirmed Cases'] = round(temp_infection_recovery_by_time_all['Recovered'] / temp_infection_recovery_by_time_all['Confirmed'], 3) * 100



temp_infection_recovery_by_time_all = temp_infection_recovery_by_time_all.melt(id_vars='Last Update', value_vars=['No. of Deaths to 100 Confirmed Cases', 

                                                                                   'No. of Recovered to 100 Confirmed Cases'], 

                                                                                   var_name='Ratio', value_name='Value')



fig = px.line(temp_infection_recovery_by_time_all, x="Last Update", y="Value", color='Ratio', 

              title='Death, Recovery Rate Over Time - All countries',color_discrete_sequence=bg)

fig.show()
temp_infection_recovery_by_time_outside_china = other_countries.groupby('Last Update').sum().reset_index()

temp_infection_recovery_by_time_outside_china.head()



# adding two more columns

temp_infection_recovery_by_time_outside_china['No. of Deaths to 100 Confirmed Cases'] = round(temp_infection_recovery_by_time_outside_china['Deaths'] / temp_infection_recovery_by_time_outside_china['Confirmed'], 3) * 100

temp_infection_recovery_by_time_outside_china['No. of Recovered to 100 Confirmed Cases'] = round(temp_infection_recovery_by_time_outside_china['Recovered'] / temp_infection_recovery_by_time_outside_china['Confirmed'], 3) * 100



temp_infection_recovery_by_time_outside_china = temp_infection_recovery_by_time_outside_china.melt(id_vars='Last Update', value_vars=['No. of Deaths to 100 Confirmed Cases', 

                                                                                   'No. of Recovered to 100 Confirmed Cases'], 

                                                                                   var_name='Ratio', value_name='Value')



fig = px.line(temp_infection_recovery_by_time_outside_china, x="Last Update", y="Value", color='Ratio', 

              title='Death, Recovery Rate Over Time - Outside China',color_discrete_sequence=bg)

fig.show()
#Runing fbprophet algorythm on confirmed cases all countries. Forecasting 15 days - All Countries.

#full_table_nc =  full_table.copy()

all_df = full_table.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

all_df = all_df[all_df['Last Update'] > '2020-01-22']



df_prophet = all_df.loc[:,["Last Update", 'Confirmed']]

df_prophet.columns = ['ds','y']

m_d = Prophet(

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

m_d.fit(df_prophet)

future_d = m_d.make_future_dataframe(periods=15)

fcst_daily = m_d.predict(future_d)

fcst_daily[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



# Plotting the predictions

fig_prpht = go.Figure()

trace1 = {

  "fill": None, 

  "mode": "markers",

  "marker_size": 10,

  "name": "Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "red"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "lightgreen"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_lower

}

trace4 = {

  "line": {"color": "blue"}, 

  "mode": "lines+markers",

  "marker_size": 5,

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat

}

data = [trace1, trace2, trace3, trace4]

layout = {

  "title": "Confirmed Cases Time Series", 

  "xaxis": {

    "title": "Date", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Confirmed Cases", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig_prpht = go.Figure(data=data, layout=layout)

fig_prpht.update_layout(template="ggplot2",title_text = '<b>Propagation Forecast - All Countries</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)

fig_prpht.update_layout(

    legend=dict(

        x=0.01,

        y=.99,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=12,

            color="Black"

        ),

        bgcolor="LightSteelBlue",

        bordercolor="Orange",

        borderwidth=2

    ))

fig_prpht.show()
#Runing fbprophet algorythm on confirmed cases Germany. Forecasting 15 days - Germany.

#full_table_nc =  full_table.copy()

germany_df = germany.groupby('Last Update')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

germany_df = germany_df[germany_df['Last Update'] > '2020-01-22']



germany_df_prophet = germany_df.loc[:,["Last Update", 'Confirmed']]

germany_df_prophet.columns = ['ds','y']

germany_m_d = Prophet(

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

germany_m_d.fit(germany_df_prophet)

germany_future_d = germany_m_d.make_future_dataframe(periods=15)

germany_fcst_daily = germany_m_d.predict(germany_future_d)

germany_fcst_daily[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



# Plotting the predictions

germany_fig_prpht = go.Figure()

trace1 = {

  "fill": None, 

  "mode": "markers",

  "marker_size": 10,

  "name": "Confirmed", 

  "type": "scatter", 

  "x": germany_df_prophet.ds, 

  "y": germany_df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "red"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": germany_fcst_daily.ds, 

  "y": germany_fcst_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "lightgreen"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": germany_fcst_daily.ds, 

  "y": germany_fcst_daily.yhat_lower

}

trace4 = {

  "line": {"color": "blue"}, 

  "mode": "lines+markers",

  "marker_size": 5,

  "name": "prediction", 

  "type": "scatter", 

  "x": germany_fcst_daily.ds, 

  "y": germany_fcst_daily.yhat

}

germany_data = [trace1, trace2, trace3, trace4]

germany_layout = {

  "title": "Confirmed Cases Time Series", 

  "xaxis": {

    "title": "Date", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Confirmed Cases", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

germany_fig_prpht = go.Figure(data=germany_data, layout=germany_layout)

germany_fig_prpht.update_layout(template="ggplot2",title_text = '<b>Propagation Forecast - Germany</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)

germany_fig_prpht.update_layout(

    legend=dict(

        x=0.01,

        y=.99,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=12,

            color="Black"

        ),

        bgcolor="LightSteelBlue",

        bordercolor="Orange",

        borderwidth=2

    ))

germany_fig_prpht.show()