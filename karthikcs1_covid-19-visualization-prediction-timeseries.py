import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})

import datetime
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
# Drop date columns if they are mostly NaN

na_columns = (confirmed_df.isna().sum() / confirmed_df.shape[0]) > 0.99

na_columns = na_columns[na_columns]



confirmed_df = confirmed_df.drop(na_columns.index, axis=1)

deaths_df = deaths_df.drop(na_columns.index, axis=1)

recoveries_df = recoveries_df.drop(na_columns.index, axis=1)
## Tidy up the data

confirmed_df = confirmed_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='confirmed')

deaths_df = deaths_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='deaths')

recoveries_df = recoveries_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='recoveries')
confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])

deaths_df['date'] = pd.to_datetime(deaths_df['date'])

recoveries_df['date'] = pd.to_datetime(recoveries_df['date'])
latest_update = confirmed_df['date'].max().strftime('%d-%B-%Y')
latest_update
full_df = confirmed_df.merge(recoveries_df).merge(deaths_df)

full_df = full_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recoveries": "Recoveries", "deaths": "Deaths"})

# Check null values

full_df.isnull().sum()
world_df = full_df.groupby(['Date']).agg({'Confirmed': ['sum'], 'Recoveries': ['sum'], 'Deaths': ['sum']}).reset_index()

world_df.columns = world_df.columns.get_level_values(0)



def add_rates(df):

    df['Confirmed Change'] = df['Confirmed'].diff()

 

    df['Mortality Rate'] = df['Deaths'] / (df['Confirmed'] - df['Confirmed Change'])

    df['Recovery Rate'] = df['Recoveries'] / (df['Confirmed'] - df['Confirmed Change'])

    df['Growth Rate'] = df['Confirmed Change'] / (df['Confirmed'] - df['Confirmed Change'])

    df['Growth Rate Change'] = df['Growth Rate'].diff()

    df['Growth Rate Accel'] = df['Growth Rate Change'] / (df['Growth Rate'] - df['Growth Rate Change'])

    return df



world_df = add_rates(world_df)
def plot_aggregate_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Confirmed'],

                             mode='lines+markers',

                             name='Confirmed',

                             line=dict(color='Yellow', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Deaths'],

                             mode='lines+markers',

                             name='Deaths',

                             line=dict(color='Red', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recoveries'],

                             mode='lines+markers',

                             name='Recoveries',

                             line=dict(color='Green', width=2)

                            ))



    return fig
plot_aggregate_metrics(world_df).show()
# Worldwide Rates

def plot_diff_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()



    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Mortality Rate'],

                             mode='lines+markers',

                             name='Mortality rate',

                             line=dict(color='red', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recovery Rate'],

                             mode='lines+markers',

                             name='Recovery rate',

                             line=dict(color='Green', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Growth Rate'],

                             mode='lines+markers',

                             name='Growth rate confirmed',

                             line=dict(color='Yellow', width=2)))

    fig.update_layout(yaxis=dict(tickformat=".2%"))

    

    return fig
fig = plot_diff_metrics(world_df)

fig.update_layout(title = 'World Metrics of Covid-19 Cases: Mortality Rate, Recovery Rate and New Cases rate')

fig.show()
# fig = go.Figure()



tmp_df = world_df.copy()

tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]



fig = px.bar(tmp_df, x='Date', y='Growth Rate Accel')

fig.update_layout(title = 'Daily Growth Rate Acceleration Worldwide', template='plotly_dark')

fig.update_yaxes(title_text='% Growth Rate Acceleration')

fig.show()
confirmed_by_country_df = full_df.groupby(['Date', 'Country']).sum().reset_index()
fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
# Log scale to allow for view

#  (1) of countries other than China, and

#  (2) identifying linear sections, which indicate exponential growth



# KCS Not showing in Log scale



# fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

# fig.update_layout(

#     template='plotly_dark',

#     yaxis_type="log"

# )

# fig.show()
by_country_df = confirmed_by_country_df.sort_values(['Country', 'Date'], ascending=[1,1])

# confirmed_by_country_df.head()
day_ctr = 0

curr_country = None 

country_100plus = []

for ind in by_country_df.index:    

    i_confirmed = by_country_df['Confirmed'][ind]

    i_country   = by_country_df['Country'][ind]

    i_date      = by_country_df['Date'][ind]

    if i_confirmed < 100:

        continue    

    if curr_country == None:

        curr_country = i_country

    elif curr_country != i_country:

        # New country is found reset the day counter 

        day_ctr = 0

        curr_country = i_country

    else:

        day_ctr += 1

        

    country_100plus.append([i_country, i_confirmed, day_ctr])

    

df_country_100plus = pd.DataFrame(country_100plus, columns =['Country', 'Confirmed', 'Days']) 
fig = px.line(df_country_100plus, x='Days', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(

    template='plotly_dark',

)



fig.update_xaxes(title_text='No. of days since 100th case')

fig.update_yaxes(title_text='No. of confirmed cases')
# Try to get unique list of countries in the df

countries = by_country_df.Country.unique()

country_df = []

for country in countries:

    country_data = by_country_df[by_country_df['Country'] == country]

    country_data = add_rates(country_data)

    country_data['Days'] = 0

    day_ctr = -1

    for ind in country_data.index:

        i_confirmed = by_country_df['Confirmed'][ind]

        if i_confirmed < 100:

            country_data['Days'][ind] = day_ctr

        else:

            day_ctr += 1

            country_data['Days'][ind] = day_ctr

    

    

    country_df.append([country,country_data])

    

country_df = pd.DataFrame(country_df, columns = ['Country','Cases'])

country_df = country_df.set_index(['Country'])
df = country_df.loc['India']['Cases']
fig = plot_aggregate_metrics(df)



fig.update_layout(title = 'Metrics of India')

fig.show()



df1 = country_df.loc['India']['Cases']

df2 = country_df.loc['Iceland']['Cases']

df3 = country_df.loc['Pakistan']['Cases']

df4 = country_df.loc['Japan']['Cases']

df5 = country_df.loc['France']['Cases']



# fig = px.line(df, x='Date', y='Growth Rate')

fig = go.Figure()

fig.add_trace(go.Scatter(x=df1['Date'], 

                             y=df1['Growth Rate'],

                             mode='lines+markers',

                             name='Growth India',

                             line=dict(color='Blue', width=2)

             ))



fig.add_trace(go.Scatter(x=df2['Date'], 

                             y=df2['Growth Rate'],

                             mode='lines+markers',

                             name='Growth Iceland',

                             line=dict(color='White', width=2)

             ))



fig.add_trace(go.Scatter(x=df3['Date'], 

                             y=df3['Growth Rate'],

                             mode='lines+markers',

                             name='Growth Pakistan',

                             line=dict(color='Green', width=2)

             ))



fig.add_trace(go.Scatter(x=df4['Date'], 

                             y=df4['Growth Rate'],

                             mode='lines+markers',

                             name='Growth Japan',

                             line=dict(color='Yellow', width=2)

             ))

fig.add_trace(go.Scatter(x=df5['Date'], 

                             y=df5['Growth Rate'],

                             mode='lines+markers',

                             name='Growth France',

                             line=dict(color='Pink', width=2)

             ))



fig.update_layout(title = 'Daily Growth Rate India', template='plotly_dark')

fig.update_yaxes(title_text=' Growth Rate ')

fig.show()


confirmed_by_country_df.groupby('Country').max().sort_values(by='Confirmed', ascending=False)[:10]