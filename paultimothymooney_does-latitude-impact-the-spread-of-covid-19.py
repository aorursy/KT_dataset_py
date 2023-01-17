# Import Python Packages

import pandas as pd

import numpy as np

import plotly.express as px

import warnings 

warnings.filterwarnings('ignore')



# Load Data

coordinates = pd.read_csv('/kaggle/input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv')

country_coordinates = coordinates[['country_code','latitude','longitude','country']]

state_coordinates = coordinates[['usa_state_code','usa_state_latitude','usa_state_longitude','usa_state']]

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df['Country/Region'].replace(['Mainland China'], 'China',inplace=True)

df['Country/Region'].replace(['US'], 'United States',inplace=True)

df['Country'] = df['Country/Region']

df = df[df.ObservationDate==np.max(df.ObservationDate)]

todays_date = '9/23/2020' # Update this line every time that you rerun the notebook



# Mortality Ratio for every country in the dataset -- technically the ratio of deaths to infections

df_deaths = pd.DataFrame(df.groupby('Country')['Deaths'].sum())

df_confirmed = pd.DataFrame(df.groupby('Country')['Confirmed'].sum())

df_confirmed['Deaths'] = df_deaths['Deaths']

df_global = df_confirmed

df_global['Mortality Ratio'] = np.round((df_global.Deaths.values/df_global.Confirmed.values)*100,2)

df_global = df_global.reset_index()

df_global = df_global.merge(country_coordinates, left_on='Country', right_on='country')

df_global = df_global[['Country','Confirmed','Deaths','Mortality Ratio','latitude','longitude','country_code']]

df_global.columns = ['Country','Confirmed','Deaths','Mortality Ratio','Latitude','Longitude','Country_Code']

df_global.to_csv('/kaggle/working/global_covid19_mortality_rates.csv')



# Mortality Ratio for every state in the USA -- technically the ratio of deaths to infections

df_usa = df[df['Country/Region']=='United States']

df_usa = df_usa[df_usa.ObservationDate==np.max(df_usa.ObservationDate)]

df_usa['State'] = df_usa['Province/State']

df_usa['Mortality Ratio'] = np.round((df_usa.Deaths.values/df_usa.Confirmed.values)*100,2)

df_usa.sort_values('Mortality Ratio', ascending= False).head(10)

df_usa = df_usa.merge(state_coordinates, left_on='State', right_on='usa_state')

df_usa['Latitude'] = df_usa['usa_state_latitude']

df_usa['Longitude'] = df_usa['usa_state_longitude']

df_usa = df_usa[['State','Confirmed','Deaths','Recovered','Mortality Ratio','Latitude','Longitude','usa_state_code']]

df_usa.columns = ['State','Confirmed','Deaths','Recovered','Mortality Ratio','Latitude','Longitude','USA_State_Code']

df_usa.to_csv('/kaggle/working/usa_covid19_mortality_rates.csv')
fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Confirmed", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,2000000],

                    title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Deaths", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,100000],

                    title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Mortality Ratio", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10],

                    title='Global COVID-19 Mortality Ratios as of '+todays_date)

fig.show()
fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:20], 

             x="Country", 

             y="Confirmed",

             title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 

             x="Country", 

             y="Deaths",

             title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 

             x="Country", 

             y="Mortality Ratio",

             title='Global COVID-19 Mortality Ratios as of '+todays_date+' for Countries with Top 20 Most Deaths')

fig.show()
fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Confirmed", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,500000],scope="usa",

                    title='USA COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Deaths", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,20000],scope="usa",

                    title='USA COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Mortality Ratio", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,10],scope="usa",

                    title='USA COVID-19 Mortality Ratios as of '+todays_date)

fig.show()
fig = px.bar(df_usa.sort_values('Confirmed',ascending=False)[0:20], 

             x="State", 

             y="Confirmed",

             title='USA COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:20], 

             x="State", 

             y="Deaths",

             title='USA COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:20], 

             x="State", 

             y="Mortality Ratio",

             title='USA COVID-19 Mortality Ratios as of '+todays_date+' for USA States with Top 20 Most Deaths')

fig.show()
df_global2 = df_global

df_global2['Latitude'] = abs(df_global2['Latitude'])

df_global2 = df_global2[df_global2['Country']!='China']



fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Confirmed",

             title='Global COVID-19 Infections vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()



fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Deaths",

             title='Global COVID-19 Deaths vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Mortality Ratio",

             title='Global COVID-19 Mortality Ratios vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

df_global.sort_values('Mortality Ratio', ascending= False).head(10)
fig = px.scatter(df_usa.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Mortality Ratio",

             title='USA States COVID-19 Mortality Ratios vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

df_usa.sort_values('Mortality Ratio', ascending= False).head(10)