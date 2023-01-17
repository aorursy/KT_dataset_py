import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objs as go

from urllib.request import urlopen

import json
df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

wv = df[df['state'] == 'West Virginia']

wv.tail()
wv_recent = wv[wv['date'] == '2020-10-14']
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)
fig = px.choropleth(wv_recent, geojson=counties, locations='fips', color='cases',

                           scope="usa",

                           labels={'cases':'Cases'},

                           color_continuous_scale="sunset",

                           hover_name = wv_recent['county'],

                           range_color=[0,wv_recent.cases.max()]

                          )

layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="Total WV COVID-19 cases by county",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
fig = px.choropleth(wv_recent, geojson=counties, locations='fips', color='deaths',

                           scope="usa",

                           labels={'deaths':'Deaths'},

                           color_continuous_scale="reds",

                           hover_name = wv_recent['county'],

                           range_color=[0,wv_recent.deaths.max()]

                          )

layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="Total WV COVID-19 deaths by county",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')

df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

df.head()
us_state_abbrev = {

'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',

'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',

'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',

'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',

'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',

'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',

'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',

'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',

'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}



df['state abb'] = df['state'].map(us_state_abbrev)
df = df.dropna()
fig = px.choropleth(locations=df['state abb'],

                    color=df["cases"], 

                    locationmode="USA-states",

                    scope="usa",

                    hover_name  = df['state'],

                    labels = {'color' : 'Cases', 'animation_frame' : 'Date'},

                    animation_frame=df['date'],

                    color_continuous_scale="sunset",

                    range_color=[0,df['cases'].max()]

                   )



layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="Total COVID-19 cases by states",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)



fig.show()
fig = px.choropleth(locations=df['state abb'],

                    color=df["deaths"], 

                    locationmode="USA-states",

                    scope="usa",

                    hover_name  = df['state'],

                    labels = {'color' : 'Deaths', 'animation_frame' : 'Date'},

                    animation_frame=df['date'],

                    color_continuous_scale="reds",

                    range_color=[0,df['deaths'].max()]

                   )



layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="Total COVID-19 deaths by states",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)



fig.show()
world = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')

world.head()
world.columns
world.continent.unique()
europe = world[world['continent'] == 'Europe']

africa = world[world['continent'] == 'Africa']
eu_df = europe[['location', 'date', 'total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million']].copy()

africa_df = africa[['location', 'date', 'total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million']].copy()



eu_df = eu_df.fillna(method='ffill')

africa_df = africa.fillna(method='ffill')



eu_df['date'] = pd.to_datetime(eu_df['date']).dt.date.astype(str)

africa_df['date'] = pd.to_datetime(africa_df['date']).dt.date.astype(str)



eu_df = eu_df.sort_values(by=['date'])

africa_df = africa_df.sort_values(by=['date'])



eu_df = eu_df[eu_df.date <= '2020-10-13']

africa_df = africa_df[africa_df.date <= '2020-10-14']





eu_df = eu_df[eu_df['total_cases'] != 0.0]

africa_df = africa_df[africa_df['total_cases'] != 0.0]
max_europe = eu_df[eu_df['location'] == 'Montenegro']['total_cases_per_million'].max()
fig = px.choropleth(locations=eu_df['location'],

                    color=eu_df["total_cases_per_million"], 

                    locationmode='country names',

                    scope="europe",

                    hover_name  = eu_df['location'],

                    labels = {'color' : 'Cases per million', 'animation_frame' : 'Date'},

                    animation_frame=eu_df['date'],

                    color_continuous_scale="purpor",

                    range_color=[0,max_europe]

                   )



layout = go.Layout(geo=dict(bgcolor= 'rgba(103, 114, 131, 1)',

                           lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="COVID-19 cases per 1 million people by country in Europe",

        x=0.45

    ),

    font=dict(size=16)

)

fig.update_layout(layout)





fig.show()
africa_max = africa_df['total_cases'].max()
fig = px.choropleth(locations=africa_df['location'],

                    color=africa_df["total_cases"], 

                    locationmode='country names',

                    scope="africa",

                    hover_name  = africa_df['location'],

                    labels = {'color' : 'Cases', 'animation_frame' : 'Date'},

                    animation_frame=africa_df['date'],

                    color_continuous_scale="sunset",

                    range_color=[0,africa_max]

                   )



layout = go.Layout(geo=dict(bgcolor= 'rgba(103, 114, 131, 1)',

                           lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="Total COVID-19 cases in Africa",

        x=0.5

    ),

    font=dict(size=16)

)

fig.update_layout(layout)





fig.show()
world_df = world[['location', 'date', 'total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million']].copy()

world_df = world_df.fillna(method='ffill')

world_df['date'] = pd.to_datetime(world_df["date"]).dt.date.astype(str)

world_df = world_df.sort_values(by=['date'])

world_df = world_df[world_df.date <= '2020-10-14']

world_df = world_df[world_df['total_cases'] != 0.0]

max_country = world_df[world_df['location'] == 'United States']['total_cases'].max()
fig = px.choropleth(locations=world_df['location'],

                    color=world_df["total_cases"], 

                    locationmode='country names',

                    scope="world",

                    hover_name  = world_df['location'],

                    labels = {'color' : 'Cases', 'animation_frame' : 'Date'},

                    animation_frame=world_df['date'],

                    color_continuous_scale="sunset",

                    range_color=[0,max_country]

                   )



layout = go.Layout(geo=dict(bgcolor= 'rgba(10,10,300,0.5)',

                           lakecolor='rgba(10,10,300,0.5)'),

    title=go.layout.Title(

        text="Total COVID-19 cases by country",

        x=0.5

    ),

    font=dict(size=16)

)

fig.update_layout(layout)





fig.show()
world_df = world_df[world_df['total_deaths'] != 0.0]

max_deaths = world_df[world_df['location'] == 'United States']['total_deaths'].max()
fig = px.choropleth(locations=world_df['location'],

                    color=world_df["total_deaths"], 

                    locationmode='country names',

                    scope="world",

                    hover_name  = world_df['location'],

                    labels = {'color' : 'Deaths', 'animation_frame' : 'Date'},

                    animation_frame=world_df['date'],

                    color_continuous_scale="reds",

                    range_color=[0,max_deaths]

                   )



layout = go.Layout(geo=dict(bgcolor= 'rgba(10,10,300,0.5)',

                           lakecolor='rgba(10,10,300,0.5)'),

    title=go.layout.Title(

        text="Total COVID-19 deaths by country",

        x=0.5

    ),

    font=dict(size=16)

)

fig.update_layout(layout)





fig.show()