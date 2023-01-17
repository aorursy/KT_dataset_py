import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objs as go

from urllib.request import urlopen

import json
df = pd.read_csv('../input/wv-counties-covid19-data/IncidRtLam_20-Oct-2020.csv')
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)
fig = px.choropleth(df, geojson=counties, locations='fips', color='dailycases_p100k_7d_avg',

                           scope="usa",

                           labels={'dailycases_p100k_7d_avg':'Daily Cases per 100k <br> (7 day average)'},

                           color_continuous_scale="reds",

                           hover_name = df['county_name'],

                           range_color=[0,df.dailycases_p100k_7d_avg.max()]

                          )

layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="WV Daily cases per county - 10/20/2020",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)

fig.update_geos(fitbounds="locations", visible=True)

fig.show()
fig = px.choropleth(df, geojson=counties, locations='fips', color='R_mean',

                           scope="usa",

                           labels={'R_mean':'R0 (mean)'},

                           color_continuous_scale="sunset",

                           hover_name = df['county_name'],

                           range_color=[0,df.R_mean.max()]

                          )

layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="WV R0 (mean) rate per county - 10/20/2020",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)

fig.update_geos(fitbounds="locations", visible=True)

fig.show()
fig = px.choropleth(df, geojson=counties, locations='fips', color='inf_potential_p100k',

                           scope="usa",

                           labels={'inf_potential_p100k':'Inf. Potential per 100k'},

                           color_continuous_scale="rdylgn_r",

                           hover_name = df['county_name'],

                           range_color=[0,df.inf_potential_p100k.max()]

                          )

layout = go.Layout(geo=dict(lakecolor= 'rgba(70, 127, 226, 1)'),

    title=go.layout.Title(

        text="WV inf. potential per county (per 100k) - 10/20/2020",

        x=0.5

    ),

    font=dict(size=16)

)



fig.update_layout(layout)

fig.update_geos(fitbounds="locations", visible=True)

fig.show()
df.columns
df[df.county_name == 'Monongalia']['population']
mon_pop = df[df.county_name == 'Monongalia']['population']

print(mon_pop)
import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(

    geojson = counties,

    locations=df['fips'], # Spatial coordinates

    z = df['dailycases_p100k_7d_avg'].astype(float), # Data to be color-coded

    colorscale = 'Reds',

    colorbar_title = "Daily Cases",

                        )

                )



fig.update_layout(

    title_text = 'West Virginia COVID-19 Daily Cases',

    geo_scope='usa',

)

'''

fig.add_trace(go.Scattergeo(

        #locationmode = 'USA-states',

        lon = [23.4642],

        lat = [91.1764],

        #text = 'Mon',

        #marker = dict(

        #    size = 3,

        #    color = 'blue',

        #    line_color='rgb(40,40,40)',

        #    line_width=0.5,

        #    sizemode = 'area'

        #            )

    )

)

'''

fig.update_geos(fitbounds="locations", visible=True)

fig.show()