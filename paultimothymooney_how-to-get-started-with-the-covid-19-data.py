# Import Python Packages

import pandas as pd

import numpy as np

import plotly.express as px

import warnings 

warnings.filterwarnings('ignore')



# Load Data

df_global = pd.read_csv('/kaggle/input/coronavirus-covid19-mortality-rate-by-country/global_covid19_mortality_rates.csv')

df_usa = pd.read_csv('/kaggle/input/coronavirus-covid19-mortality-rate-by-country/usa_covid19_mortality_rates.csv')

todays_date = '9/09/2020' # Update this line every time that you rerun the notebook
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

                    title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Deaths", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,20000],scope="usa",

                    title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Mortality Ratio", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,10],scope="usa",

                    title='Global COVID-19 Mortality Ratios as of '+todays_date)

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