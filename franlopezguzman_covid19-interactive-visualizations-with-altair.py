import numpy as np

import pandas as pd

import altair as alt
df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'], index_col='Id')

df.head()
df.rename(columns={'Date': 'date',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)
df_countries = df.drop(['state'], axis=1).groupby(['country','date']).sum().reset_index()

df_countries.head()
#df_countries['country'].unique()

countries_list = ['China', 'Korea, South', 'Italy', 'Iran', 'Spain', 'US',  'Brazil']



df_countries[df_countries['country'].isin(countries_list)].sort_values('date', ascending=False).head()
base = alt.Chart(

    df_countries[df_countries['country'].isin(countries_list)]

).mark_area(

    interpolate = 'monotone',

    fillOpacity = .8

).encode(

    x = 'date',

    y = 'confirmed',

    color = 'country'

)



base.encode(y='confirmed') & base.encode(y='deaths')
interval = alt.selection_interval(encodings=['x'])

color = alt.condition(interval, 'country', alt.value('lightgray'))



point_base = alt.Chart(

    df_countries[df_countries['country'].isin(countries_list)]

).mark_point().encode(

    x = 'date',

    color = color

).properties(

    selection = interval

)



hist_base = alt.Chart(

    df_countries[df_countries['country'].isin(countries_list)]

).mark_bar().encode(

    y = 'country',

    color = 'country'

).transform_filter(interval)





point_confirmed = point_base.encode(y='confirmed')

point_deaths = point_base.encode(y='deaths')



hist_confirmed = hist_base.encode(x='confirmed')

hist_deaths = hist_base.encode(x='deaths')



(point_confirmed & hist_confirmed) & (point_deaths & hist_deaths)
sort_order = ['China', 'Korea, South', 'Italy', 'Iran', 'Spain', 'US', 'Brazil']



step = 50

overlap = 3



alt.Chart(

    df_countries[df_countries['country'].isin(countries_list)], 

    height=step,

    width = 12*step,

).mark_area(

    interpolate = 'monotone',

    fillOpacity = .8,

).encode(

    alt.X('date'),

    alt.Y('confirmed:Q',

          scale=alt.Scale(range=[step, -step * overlap]),

          axis=None),

    alt.Fill('country'),

    tooltip = ['date','country','confirmed','deaths']

).facet(

    row = alt.Row('country:N',

                  sort=sort_order, #alt.EncodingSortField(field='confirmed', order='descending'),

                  title=None,

                  header=alt.Header(labelAngle=0, labelAlign='right'))

).properties(

    bounds = 'flush',

    title = 'Evolution of onfirmed cases per country'

).configure_facet(

    spacing=0

).configure_view(

    stroke=None

)
df_countries.sort_values(by=['country','date'])

df_countries['daily_confirmed'] = df_countries.groupby('country')['confirmed'].diff().fillna(0)

df_countries['daily_deaths'] = df_countries.groupby('country')['deaths'].diff().fillna(0)
sort_order = ['China', 'Korea, South', 'Italy', 'Iran', 'Spain', 'US', 'Brazil']



step = 50

overlap = 3



alt.Chart(

    df_countries[df_countries['country'].isin(countries_list)], 

    height=step,

    width = 12*step,

).mark_area(

    interpolate = 'monotone',

    fillOpacity = .8,

).encode(

    alt.X('date'),

    alt.Y('daily_confirmed:Q',

          scale=alt.Scale(range=[step, -step * overlap]),

          axis=None),

    alt.Fill('country'),

    tooltip = ['date','country','daily_confirmed','daily_deaths']

).facet(

    row = alt.Row('country:N',

                  sort=sort_order, #alt.EncodingSortField(field='confirmed', order='descending'),

                  title=None,

                  header=alt.Header(labelAngle=0, labelAlign='right'))

).properties(

    bounds = 'flush',

    title = 'Evolution of daily confirmed cases per country',

).configure_facet(

    spacing=0

).configure_view(

    stroke=None

)
us = df[df['country']=='US'].groupby(['state','date']).sum().reset_index()

us = us[us['date']>='2020-03-07']

relevant_states = us.sort_values('confirmed', ascending=False)['state'].unique()[:20].tolist()

relevant_states.remove('New York')



us['daily_confirmed'] = us.groupby('state')['confirmed'].diff().fillna(0)

us['daily_deaths'] = us.groupby('state')['deaths'].diff().fillna(0)



us['is_NY'] = us['state'] == 'New York'



alt.Chart(

    us.groupby(['date', 'is_NY']).sum().reset_index()

).mark_area(

    interpolate = 'monotone',

    fillOpacity = .8,

).encode(

    x = 'date',

    y = 'confirmed',

    color = 'is_NY:N',

    tooltip = ['date','confirmed']

).properties(

    title='Confirmed cases in NY and rest of USA',

)
us.drop('is_NY', axis=1, inplace=True)



step = 50

overlap = 3



alt.Chart(

    us[us['state'].isin(relevant_states)], 

    height=step,

    width = 12*step,

).mark_area(

    interpolate = 'monotone',

    fillOpacity = .8,

).encode(

    alt.X('date'),

    alt.Y('confirmed:Q',

          scale=alt.Scale(range=[step, -step * overlap]),

          axis=None),

    alt.Fill('state', legend=None),

    tooltip = ['date','state','confirmed','deaths']

).facet(

    row = alt.Row('state:N',

                  sort=relevant_states,

                  title=None,

                  header=alt.Header(labelAngle=0, labelAlign='right'))

).properties(

    bounds = 'flush',

    title = 'Evolution of confirmed cases per state'

).configure_facet(

    spacing=0

).configure_view(

    stroke=None

)