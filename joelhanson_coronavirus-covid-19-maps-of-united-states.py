!pip install chart_studio

!pip install plotly-geo
import pandas as pd

import numpy as np

import json



import plotly.offline as py

import plotly.express as px

import plotly.figure_factory as ff



with open('../input/geojson-counties-fips/geojson-counties-fips.json') as response:

    counties = json.load(response)
us_covid_df = pd.read_csv("../input/coronavirus-covid19-data-in-the-united-states/us-states.csv")

us_covid_df.head()
us_covid_country_wise_df = pd.read_csv("../input/coronavirus-covid19-data-in-the-united-states/us-counties.csv")

agg_us_covid_country_wise_df = us_covid_country_wise_df[['state', 'county', 'cases', 'deaths', 'fips']].groupby(by=['state', 'fips', 'county'], as_index=False).sum()

us_covid_country_wise_df.head()
state_to_st_dict = {

    "Alabama": "AL",

    "Alaska": "AK",

    "American Samoa": "AS",

    "Arizona": "AZ",

    "Arkansas": "AR",

    "California": "CA",

    "Colorado": "CO",

    "Northern Mariana Islands": "MP",

    "Connecticut": "CT",

    "Delaware": "DE",

    "District of Columbia": "DC",

    "Florida": "FL",

    "Georgia": "GA",

    "Guam": "GU",

    "Hawaii": "HI",

    "Idaho": "ID",

    "Illinois": "IL",

    "Indiana": "IN",

    "Iowa": "IA",

    "Kansas": "KS",

    "Kentucky": "KY",

    "Louisiana": "LA",

    "Maine": "ME",

    "Maryland": "MD",

    "Massachusetts": "MA",

    "Michigan": "MI",

    "Minnesota": "MN",

    "Mississippi": "MS",

    "Missouri": "MO",

    "Montana": "MT",

    "Nebraska": "NE",

    "Nevada": "NV",

    "New Hampshire": "NH",

    "New Jersey": "NJ",

    "New Mexico": "NM",

    "New York": "NY",

    "North Carolina": "NC",

    "North Dakota": "ND",

    "Ohio": "OH",

    "Oklahoma": "OK",

    "Oregon": "OR",

    "Pennsylvania": "PA",

    "Puerto Rico": "",

    "Rhode Island": "RI",

    "South Carolina": "SC",

    "South Dakota": "SD",

    "Tennessee": "TN",

    "Texas": "TX",

    "Virgin Islands": "VI",

    "Utah": "UT",

    "Vermont": "VT",

    "Virginia": "VA",

    "Washington": "WA",

    "West Virginia": "WV",

    "Wisconsin": "WI",

    "Wyoming": "WY",

}

us_covid_df["state-code"] = us_covid_df.state.apply(lambda state: state_to_st_dict.get(state))

agg_us_covid_df = us_covid_df[['state', 'state-code', 'cases', 'deaths']].groupby(by=['state', 'state-code'], as_index=False).sum()
fig = px.choropleth(agg_us_covid_df, geojson=counties, locations='state-code', color='cases', locationmode='USA-states',

                           scope="usa",

                           hover_name="state",

                           labels={'cases':'Number of cases', 'deaths': "Number of deaths"}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(agg_us_covid_df, geojson=counties, locations='state-code', color='deaths', locationmode='USA-states',

                           scope="usa",

                           color_continuous_scale="OrRd",

                           hover_name="state",

                           labels={'cases':'Number of cases', 'deaths': "Number of deaths"}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(agg_us_covid_country_wise_df, geojson=counties, locations='fips', color='cases',

                           range_color=(0, 12),

                           scope="usa",

                           hover_name="county",

                           labels={'cases':'Number of cases', 'deaths': "Number of deaths"}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(agg_us_covid_country_wise_df.loc[agg_us_covid_country_wise_df['deaths'] > 0], geojson=counties, locations='fips', color='deaths',

                           color_continuous_scale="OrRd",

                           range_color=(12, 0),

                           scope="usa",

                           hover_name="county",

                           labels={'cases':'Number of cases', 'deaths': "Number of deaths"}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
def state_wise_maps(state, state_df):

    fig = px.choropleth(state_df, geojson=counties, color="cases", scope="usa", hover_name="county",

                    locations="county", featureidkey="county",

                    projection="mercator", title=state

                   )

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.show()
states = agg_us_covid_country_wise_df.state.unique().tolist()

for state in states:

    state_wise_maps(state, agg_us_covid_country_wise_df.loc[agg_us_covid_country_wise_df['state'] == state])