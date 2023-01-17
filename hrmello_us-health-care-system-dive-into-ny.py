import numpy as np

import pandas as pd



covid_stats = pd.read_csv("/kaggle/input/uncover/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv", dtype={"fips": str})

health_rankings = pd.read_csv("/kaggle/input/uncover/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv")

icu_facilities = pd.read_csv("/kaggle/input/uncover/hifld/hifld/urgent-care-facilities.csv")
print("COVID STATS COLUMNS:")

print(covid_stats.keys())

print("========================================================================")

print("HEALTH RANKINGS COLUMNS:")

print(health_rankings.keys())

print("========================================================================")

print("ICU FACILITIES COLUMNS:")

print(icu_facilities.keys())

print("========================================================================")
for key in health_rankings.keys():

    print(key)
columns = ["fips", "state", "county", "num_deaths", "years_of_potential_life_lost_rate", 

          "percent_fair_or_poor_health", "average_number_of_physically_unhealthy_days",

          "average_number_of_mentally_unhealthy_days", "percent_smokers", "num_primary_care_physicians",

          "preventable_hospitalization_rate", "num_unemployed", "labor_force", "income_ratio", "num_households",

          "overcrowding", "life_expectancy"]



health_rankings_selected = health_rankings[columns]
covid_stats.fillna(0)
import plotly.graph_objects as go



covid_stats_last = covid_stats[covid_stats["date"] == "2020-03-30"].copy()



fig = go.Figure(data=go.Choropleth(

    locations=covid_stats_last['state'], # Spatial coordinates

    z = covid_stats_last['positive'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = "Reds",

    colorbar_title = "# positives",

))



fig.update_layout(

    title_text = 'Number of positive tests per US state',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
icus_per_state = pd.DataFrame(icu_facilities.groupby(["state"])["id"].nunique()).reset_index()



us_census = pd.read_csv("../input/us-census-demographic-data/acs2017_county_data.csv")



# abbreviation and full state name csv

states_abbrev = pd.read_csv("https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv")



# merging the abbreviations within the census dataset 

us_census = us_census.merge(states_abbrev, on = "State")

us_census = us_census.rename(columns = {"State": "fullState", "Abbreviation": "state"})



#population per state

popState = us_census.groupby("state")["TotalPop"].sum().reset_index()



icupopdf = icus_per_state.merge(popState, on = "state")



icupopdf["ICUPopRatio"] = icupopdf["id"]/icupopdf["TotalPop"] * 100000



fig = go.Figure(data=go.Choropleth(

    locations=icupopdf['state'], # Spatial coordinates

    z = icupopdf['ICUPopRatio'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = "Blues",

    colorbar_title = "# of ICUs per 100k people",

))



fig.update_layout(

    title_text = 'ICU density per state',

    geo_scope='usa', # limite map scope to USA

)



fig.show()

import plotly.express as px



fig = px.line(covid_stats[covid_stats["state"] == "NY"], x="date", y="positiveincrease", title='Daily increase in COVID-19 cases')

fig.show()
fig = px.line(covid_stats[covid_stats["state"] == "NY"], x="date", y="positive", title='Total increase in COVID-19 cases in New York')

fig.show()
covid_ny = covid_stats[covid_stats["state"] == "NY"].copy()



fig = go.Figure()

fig.add_trace(go.Scatter(x=covid_ny["date"], y=covid_ny["hospitalized"],

                    mode='lines',

                    name='Hospitalized'))

fig.add_trace(go.Scatter(x=covid_ny["date"], y=covid_ny["death"],

                    mode='lines',

                    name='Deaths'))