import numpy as np 

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime



%matplotlib inline
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
countries_df = pd.read_csv('../input/countries-of-the-world/countries of the world.csv')

countries_df['Country'] = countries_df['Country'].str.strip()

countries_df['Country'] = countries_df['Country'].replace('United States', 'US')
us_state_populations = pd.read_csv('../input/us-states-population/population-us-states.csv')

us_state_populations = us_state_populations.rename(columns={'POPESTIMATE2019':'Population', 'NAME': 'State'})

us_state_populations = us_state_populations[['State', 'Population']]
# filter out China and cruise-ships and focus on 2nd wave of countries impacted

# this is not a complete list and is curated manually



world_confirmed_df = confirmed_df[~confirmed_df['Country/Region'].str.contains("China")]

most_impacted_countries = world_confirmed_df.sort_values('3/19/20', ascending=False).head(20)['Country/Region']



countries = ['Japan', 'South Korea', 'Iran', 'Italy', 'France', 'Germany', 'Spain', 'US', 'UK', 'Netherlands', 'Sweden', 'Belgium']



world_confirmed_df = confirmed_df[confirmed_df['Country/Region'].isin(countries) | confirmed_df['Country/Region'].isin(most_impacted_countries)]



def cleanup(df):

    # drop unused columns before converting from wide to long format

    del df['Lat']

    del df['Long']



    df = df.melt(id_vars=['Country/Region', 'Province/State'], var_name='Date', value_name='Count')

    df = df.sort_values(by=['Country/Region', 'Province/State'])



    # convert into propert datetime and simple location string

    df['Datetime'] = pd.to_datetime(df['Date'], format='%m/%d/%y')



    # aggregate US from city data

    us_df = df[df['Country/Region'] == 'US'].groupby(['Country/Region', 'Datetime']).sum().reset_index()



    df = pd.concat([df, us_df], sort=True)



    # merge country + city names into one column

    df['Location'] = np.where(df['Province/State'].isnull(), df['Country/Region'], df['Country/Region'] + " / "  + df['Province/State'])



    # no longer need original columns

    del df['Country/Region']

    del df['Province/State']

    del df['Date']

    return df



world_confirmed_df = cleanup(world_confirmed_df)
countries_confirmed_df = world_confirmed_df[~world_confirmed_df['Location'].str.contains("/")]



countries_confirmed_df = countries_confirmed_df.merge(countries_df, left_on='Location', right_on='Country')

countries_confirmed_df['Percent'] = 100 * countries_confirmed_df['Count'] / countries_confirmed_df['Population']



countries_confirmed_df = countries_confirmed_df.sort_values(by=['Location','Datetime'])



fig = px.line(countries_confirmed_df, x="Datetime", y="Percent", color="Location",

              height=600,

              line_dash_map={'US': 'solid'},

              line_dash_sequence=['dot'],

              line_dash='Location',

              line_shape='spline',

              render_mode='svg'

             )

fig.update_layout(legend=dict(x=0, y=-.25, traceorder="normal", orientation="h"))

fig.update_traces(mode='lines+markers')

fig.show()
countries_started_df = countries_confirmed_df[countries_confirmed_df['Count'] > 100]



countries_start_dates = countries_started_df.groupby('Location').agg(StartDatetime=('Datetime', 'min'))

countries_started_df = countries_started_df.merge(countries_start_dates, on='Location')



countries_started_df['nth_day'] = (countries_started_df['Datetime'] - countries_started_df['StartDatetime']).dt.days



fig = px.line(countries_started_df, x="nth_day", y="Count", color="Location",

              height=600,

              line_dash_map={'US': 'solid'},

              line_dash_sequence=['dot'],

              line_dash='Location',

              line_shape='spline',

             )

fig.update_layout(legend=dict(x=0, y=-.25, traceorder="normal", orientation="h"))

fig.update_traces(mode='lines+markers')

fig.show()
countries_started_df = countries_confirmed_df[countries_confirmed_df['Count'] > 100]



countries_start_dates = countries_started_df.groupby('Location').agg(StartDatetime=('Datetime', 'min'))

countries_started_df = countries_started_df.merge(countries_start_dates, on='Location')



countries_started_df['nth_day'] = (countries_started_df['Datetime'] - countries_started_df['StartDatetime']).dt.days



fig = px.line(countries_started_df, x="nth_day", y="Percent", color="Location",

              height=600,

              line_dash_map={'US': 'solid'},

              line_dash_sequence=['dot'],

              line_dash='Location',

              line_shape='spline',

             )

fig.update_layout(legend=dict(x=0, y=-.25, traceorder="normal", orientation="h"))

fig.update_traces(mode='lines+markers')

fig.show()