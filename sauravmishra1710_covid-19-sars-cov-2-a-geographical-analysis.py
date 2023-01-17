# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the necessary libraries



import numpy as np 

import pandas as pd

from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown

import plotly.graph_objs as go

import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.express as px

from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs

import plotly as ply

import pycountry

import folium 

from folium import plugins

import json

from folium.plugins import HeatMap, HeatMapWithTime





%config InlineBackend.figure_format = 'retina'

init_notebook_mode(connected=True)





# Utility Functions



'''Display markdown formatted output like bold, italic bold etc.'''

def formatted_text(string):

    display(Markdown(string))





'''highlight the maximum in a Series or DataFrame'''  

def highlight_max(data, color='yellow'):

    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  # from .apply(axis=None)

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)   
covid_19 = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
print("Covid_19 Shape", covid_19.shape)
covid_19.head()
formatted_text('***Covid 19 data information -***')

covid_19.info()
formatted_text('***NULL values in the data -***')

covid_19.isnull().sum()
# Convert 'Last Update' column to datetime object

covid_19['Date'] = covid_19['Date'].apply(pd.to_datetime)



# Fill the missing values in 'Province/State' with the 'Country' name.

covid_19['Province/State'] = covid_19['Province/State'].replace(np.nan, covid_19['Country/Region'])



# Fill the missing values (if any) in 'Confirmed', 'Deaths', 'Recovered' with the 0

covid_19['Confirmed'] = covid_19['Confirmed'].replace(np.nan, 0)

covid_19['Deaths'] = covid_19['Deaths'].replace(np.nan, 0)

covid_19['Recovered'] = covid_19['Recovered'].replace(np.nan, 0)





# Lets rename the columns - 'Province/State' and 'Last Update' to remove the '/' and space respectively.

covid_19.rename(columns={'Country/Region': 'Country', 'Province/State': 'State'}, inplace=True)



# Convert 'Mainland China' to 'China'

covid_19['Country'] = np.where(covid_19['Country'] == 'Mainland China', 'China', covid_19['Country'])



# Data Glimpse

covid_19.head()
# Check for the missing values again to ensure that there are no more remaining

formatted_text('***NULL values in the data -***')

covid_19.isnull().sum()
# Lets check the total #Countries affected by nCoV



formatted_text('***Affected Countries -***')

Covid_19_Countries = covid_19['Country'].unique().tolist()

print(Covid_19_Countries)

print("\n------------------------------------------------------------------")

print("\nTotal countries affected by nCoV: ",len(Covid_19_Countries))
# Now lets see the Country - 'Others' which is there in the list above

formatted_text('***Affected Country - Others***')

covid_19[covid_19['Country'] == 'US'].head()
# Lets create a subset of the data for the cruise ship



diamond_cruise_ship_cases = covid_19[covid_19['Country'] == 'Cruise Ship']



# Data Glimpse

diamond_cruise_ship_cases.head()
# Now that we have created a different subset for the cruise ship data, lets derive a subset with only the country data

covid_19_world_data = covid_19[covid_19['Country'] != 'Cruise Ship']



formatted_text('***World Data -***')

# Data Glimpse

covid_19_world_data.head()
formatted_text('***World Data Countries Afftected -***')



print(covid_19_world_data.Country.unique().tolist())

print("\nTotal number of countries: ", len(covid_19_world_data.Country.unique().tolist()))
formatted_text('***Country and State wise grouped data -***')



covid_19_country_wise_data = covid_19_world_data.groupby(['Country', 'State'])['Confirmed', 'Deaths', 'Recovered'].max()

covid_19_country_wise_data
strDate = covid_19_world_data['Date'][-1:].astype('str')

year = int(strDate.values[0].split('-')[0])

month = int(strDate.values[0].split('-')[1])

day = int(strDate.values[0].split('-')[2].split()[0])



formatted_text('***Last reported case date-time***')

print(strDate)

print(year)

print(month)

print(strDate.values[0].split('-')[2].split())
latest_covid_19_data = covid_19_world_data[covid_19_world_data['Date'] == pd.Timestamp(date(year,month,day))]



latest_covid_19_data.reset_index(inplace=True, drop=True)



latest_covid_19_data.head()
latest_covid_19_data[pd.isnull(latest_covid_19_data).any(axis=1)]
CountryWiseData = pd.DataFrame(latest_covid_19_data.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum())

CountryWiseData['Country'] = CountryWiseData.index

CountryWiseData.index = np.arange(1, len(covid_19_world_data.Country.unique().tolist())+1)



CountryWiseData = CountryWiseData[['Country','Confirmed', 'Deaths', 'Recovered']]



formatted_text('***Country wise numbers of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')



#CountryWiseData = pd.merge(latest_covid_19_data[['Country', 'Lat','Long']], CountryWiseData, on='Country')



#CountryWiseData = CountryWiseData.drop_duplicates(subset = "Country", keep = 'first', inplace = True) 



CountryWiseData
# Import the WORLD Latitute Longitude Data



world_lat_lon_coordinates = pd.read_csv('/kaggle/input/world-coordinates/world_coordinates.csv')

world_lat_lon_coordinates.head()
# Merge the Country co-coordinates above to the country wise data we created.



CountryWiseData = pd.merge(world_lat_lon_coordinates, CountryWiseData, on='Country')

CountryWiseData.head()
WorldMap = folium.Map(location=[0,0], zoom_start=1.5,tiles='cartodbpositron')



formatted_text('***Click on the pin to veiw details stats***')



for lat, long, confirmed, deaths, recovered, country in zip(CountryWiseData['latitude'],

                                                           CountryWiseData['longitude'],

                                                           CountryWiseData['Confirmed'],

                                                           CountryWiseData['Deaths'],

                                                           CountryWiseData['Recovered'], 

                                                           CountryWiseData['Country']):



    if (deaths == 0):

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               '<strong>Country:</strong> ' + str(country) + '<br>'

                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'

                    , tooltip = str(country), fill_color='rgb(55, 83, 109)').add_to(WorldMap)



    else:

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               '<strong>Country:</strong> ' + str(country) + '<br>'

                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

                    , tooltip = str(country), fill_color='rgb(26, 118, 255)').add_to(WorldMap)

        

WorldMap
WorldMap = folium.Map(location=[0,0], zoom_start=1.5,tiles='Stamen Toner')



formatted_text('***Click on the dots to veiw details stats***')



for lat, long, confirmed, deaths, recovered, country, state in zip(latest_covid_19_data['Lat'],

                                                           latest_covid_19_data['Long'],

                                                           latest_covid_19_data['Confirmed'],

                                                           latest_covid_19_data['Deaths'],

                                                           latest_covid_19_data['Recovered'], 

                                                           latest_covid_19_data['Country'],

                                                           latest_covid_19_data['State']):



    if (deaths == 0):

        folium.CircleMarker(location=[lat, long]

                    , radius=3

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               '<strong>Country:</strong> ' + str(country) + '<br>'

                               '<strong>State:</strong> ' + str(state) + '<br>'

                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , color='blue'

                    , tooltip = str(state)

                    , fill_color='blue'

                    , fill_opacity=0.7).add_to(WorldMap)



    else:

        folium.CircleMarker(location=[lat, long]

                    , radius=3

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               '<strong>Country:</strong> ' + str(country) + '<br>'

                               '<strong>State:</strong> ' + str(state) + '<br>'

                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , color='red'

                    , tooltip = str(state)

                    , fill_color='red'

                    , fill_opacity=0.7).add_to(WorldMap)

        

WorldMap
choropleth_map_confirmed = px.choropleth(CountryWiseData, locations='Country', 

                    locationmode='country names', color='Confirmed', 

                    hover_name='Country', range_color=[1,max(CountryWiseData.Confirmed)], 

                    color_continuous_scale='reds', 

                    title='Covid-19 Globally Confirmed Countries')



choropleth_map_confirmed.update(layout_coloraxis_showscale=False)

iplot(choropleth_map_confirmed)
choropleth_map_deaths = px.choropleth(CountryWiseData, locations='Country', locationmode='country names', color='Deaths', hover_name='Country', range_color=[1,max(CountryWiseData.Deaths)], 

                                      color_continuous_scale='reds', title='Covid-19 Global Deaths Numbers')



choropleth_map_deaths.update(layout_coloraxis_showscale=False)

iplot(choropleth_map_deaths)
choropleth_map_recovered = px.choropleth(CountryWiseData, locations='Country', 

                    locationmode='country names', color='Recovered', 

                    hover_name='Country', range_color=[1,max(CountryWiseData.Recovered)], 

                    color_continuous_scale='reds', 

                    title='Covid-19 Global Recovered Cases')



choropleth_map_recovered.update(layout_coloraxis_showscale=False)

iplot(choropleth_map_recovered)
chinese_data_over_time = covid_19[(covid_19['Country'] == 'China')]

chinese_data_over_time.head()
china_statewise_data = chinese_data_over_time.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max()



china_statewise_data['State'] = china_statewise_data.index

china_statewise_data.index = np.arange(1, len(china_statewise_data.State.unique().tolist())+1)



china_statewise_data = china_statewise_data[['State','Confirmed', 'Deaths', 'Recovered']]



formatted_text('***Country wise numbers of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')



china_statewise_data.head()
# Extract the state latitude and longitude coordinates from the time series data.

china_coordinates = chinese_data_over_time[['State','Lat','Long']]

china_coordinates.drop_duplicates(keep='first', inplace=True)



china_coordinates.index = np.arange(1, len(china_coordinates.State.unique().tolist())+1)



china_coordinates.head()
china_statewise_data = pd.merge(china_coordinates, china_statewise_data, on='State')



china_statewise_data.head()
china_lat = 35.8617

china_lon = 104.1954



formatted_text('***Click on the pin to veiw details stats***')



ChinaMap = folium.Map(location=[china_lat, china_lon], zoom_start=4, tiles='cartodbpositron')



for lat, long, confirmed, deaths, recovered, state in zip(china_statewise_data['Lat'],

                                                           china_statewise_data['Long'],

                                                           china_statewise_data['Confirmed'],

                                                           china_statewise_data['Deaths'],

                                                           china_statewise_data['Recovered'], 

                                                           china_statewise_data['State']):

    

    if (deaths == 0):

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(55, 83, 109)').add_to(ChinaMap)

    else:

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(26, 118, 255)').add_to(ChinaMap)

    

    

ChinaMap
# Load the CHina geo json file



with open('/kaggle/input/china-geo-json/china_geojson.json') as file:

    china = json.load(file)
china_conf_choropleth = go.Figure(go.Choroplethmapbox(geojson=china, locations=china_statewise_data['State'],

                                                      z=china_statewise_data['Confirmed'], colorscale='Aggrnyl',

                                                      zmin=0, zmax=max(china_statewise_data['Confirmed']), marker_opacity=0.5, marker_line_width=0))



china_conf_choropleth.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, 

                                    mapbox_center = {"lat": china_lat, "lon": china_lon})



china_conf_choropleth.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



iplot(china_conf_choropleth)
china_death_choropleth = go.Figure(go.Choroplethmapbox(geojson=china, locations=china_statewise_data['State'],

                                                      z=china_statewise_data['Deaths'], colorscale='Sunset',

                                                      zmin=0, zmax=max(china_statewise_data['Deaths']), marker_opacity=0.5, marker_line_width=0))



china_death_choropleth.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, 

                                    mapbox_center = {"lat": china_lat, "lon": china_lon})



china_death_choropleth.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



iplot(china_death_choropleth)
china_recovered_choropleth = go.Figure(go.Choroplethmapbox(geojson=china, locations=china_statewise_data['State'],

                                                      z=china_statewise_data['Recovered'], colorscale='Brbg',

                                                      zmin=0, zmax=10000, marker_opacity=0.5, marker_line_width=0))



china_recovered_choropleth.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, 

                                    mapbox_center = {"lat": china_lat, "lon": china_lon})



china_recovered_choropleth.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



iplot(china_recovered_choropleth)
rest_of_world = CountryWiseData[CountryWiseData['Country'] != 'China']

rest_of_world.head()
rest_of_world_confirmed = px.choropleth(rest_of_world, locations='Country', 

                    locationmode='country names', color='Confirmed', 

                    hover_name='Country', range_color=[1, 10000], 

                    color_continuous_scale='Geyser', 

                    title='Covid-19 Rest of World Confirmed Cases')



iplot(rest_of_world_confirmed)
rest_of_world_death = px.choropleth(rest_of_world, locations='Country', 

                    locationmode='country names', color='Deaths', 

                    hover_name='Country', range_color=[0, len(rest_of_world.Deaths)], 

                    color_continuous_scale='Picnic', 

                    title='Covid-19 Rest of World Death Cases')



iplot(rest_of_world_death)
rest_of_world_recovered = px.choropleth(rest_of_world, locations='Country', 

                    locationmode='country names', color='Recovered', 

                    hover_name='Country', range_color=[1,len(rest_of_world.Recovered)], 

                    color_continuous_scale='viridis', 

                    title='Covid-19 Rest of World Recovered Cases')



iplot(rest_of_world_recovered)
formatted_text('***Countries withh all reported cases recovered -***')

print(rest_of_world[rest_of_world['Confirmed'] == 

                    rest_of_world['Recovered']][['Country','Confirmed', 'Recovered']].reset_index())
# diamond_cruise_ship_cases.reset_index(drop=True, inplace=True)



# # We only need the latest data here

# temp_ship = diamond_cruise_ship_cases.sort_values(by='Date', ascending=False).head(1)[['State', 'Confirmed', 

#                                                                                        'Deaths', 'Recovered']]



# temp_ship
# formatted_text('***Click on the pin to veiw details stats***')

# cruiseMap = folium.Map(location=[diamond_cruise_ship_cases.iloc[0]['Lat'], diamond_cruise_ship_cases.iloc[0]['Long']], 

#                        tiles='cartodbpositron', min_zoom=8, max_zoom=12, zoom_start=12)



# folium.Marker(location=[diamond_cruise_ship_cases.iloc[0]['Lat'], diamond_cruise_ship_cases.iloc[0]['Long']],

#         popup =   '<strong>Ship : ' + str(temp_ship.iloc[0]['State']) + '<br>' +

#                     '<strong>Confirmed : ' + str(temp_ship.iloc[0]['Confirmed']) + '<br>' +

#                     '<strong>Deaths : ' + str(temp_ship.iloc[0]['Deaths']) + '<br>' +

#                     '<strong>Recovered : ' + str(temp_ship.iloc[0]['Recovered'])

#                     , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

#                     , tooltip = str(temp_ship.iloc[0]['State']), fill_color='rgb(26, 118, 255)').add_to(cruiseMap)



# cruiseMap
china_statewise_data["Country"] = "China" # in order to have a single root node



fig1 = px.treemap(china_statewise_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country", "State"], values="Confirmed", title='Number of Confirmed Cases in Chinese Provinces',

                 color_discrete_sequence = px.colors.qualitative.Prism, hover_data=["Confirmed"])



fig1.data[0].textinfo = 'label+text+value+percent entry'

py.offline.iplot(fig1)



fig2 = px.treemap(china_statewise_data.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country", "State"], values="Deaths", title='Number of Deaths Reported in Chinese Provinces',

                 color_discrete_sequence = px.colors.qualitative.Prism, hover_data=["Deaths"])



fig2.data[0].textinfo = 'label+text+value+percent entry'

py.offline.iplot(fig2)
temp2 = pd.DataFrame(covid_19_world_data.groupby(['Country', 'State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())

temp2['Global'] = "Global" # to have a single root
fig = px.treemap(temp2, 

                 path=["Global", "Country"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases Around The Globe',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()



fig = px.treemap(temp2, 

                 path=["Global", "Country", "State"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases Around The Globe',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent parent+percent entry'

fig.show()

fig = px.treemap(temp2, 

                 path=["Global", "Country"], values="Deaths", height=700,

                 title='Number of Deaths reported Globally',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()



fig = px.treemap(temp2, 

                 path=["Global", "Country", "State"], values="Deaths", height=700,

                 title='Number of Deaths reported Globally',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent parent+percent entry'

fig.show()
conf_heatmap = folium.Map(location=[0,0], zoom_start=2)



HeatMap(data=latest_covid_19_data[['Lat', 'Long', 'Confirmed']].groupby(['Lat', 'Long']).sum().reset_index().values.tolist(),radius=18, max_zoom=12).add_to(conf_heatmap)



conf_heatmap
deaths_heatmap = folium.Map(location=[0,0], zoom_start=2)



HeatMap(data=latest_covid_19_data[['Lat', 'Long', 'Deaths']].groupby(['Lat', 'Long']).sum().reset_index().values.tolist(),radius=18, max_zoom=12).add_to(deaths_heatmap)



deaths_heatmap
Italy_Covid19 = pd.read_csv('/kaggle/input/coronavirus-in-italy/dati-regioni/dpc-covid19-ita-regioni-20200318.csv')

Italy_Covid19.head()
Italy_Covid19[['denominazione_regione','totale_positivi','deceduti']].sort_values('totale_positivi', ascending=False).head(5)
italy_lat = 42.50

italy_lon = 12.50



formatted_text('***Click on the pin to veiw details stats***')



ItalyMap = folium.Map(location=[italy_lat, italy_lon], zoom_start=6, tiles='cartodbpositron')



for lat, long, confirmed, current_positive, deaths, recovered, region in zip(Italy_Covid19['lat'],

                                                           Italy_Covid19['long'],

                                                           Italy_Covid19['totale_casi'], 

                                                           Italy_Covid19['totale_positivi'],

                                                           Italy_Covid19['deceduti'],

                                                           Italy_Covid19['ricoverati_con_sintomi'], 

                                                           Italy_Covid19['denominazione_regione']):

    

    if (deaths == 0):

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(region).capitalize() + '<br>'

                                 '<strong>Total Confirmed:</strong> ' + str(confirmed) + '<br>'

                                 '<strong>Current Positive Cases:</strong> ' + str(int(current_positive)) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'

                    , tooltip = str(region).capitalize(), fill_color='rgb(55, 83, 109)').add_to(ItalyMap)

    else:

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(region).capitalize() + '<br>'

                                 '<strong>Total Confirmed:</strong> ' + str(confirmed) + '<br>'

                                 '<strong>Current Positive Cases</strong> ' + str(int(current_positive)) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

                    , tooltip = str(region).capitalize(), fill_color='rgb(26, 118, 255)').add_to(ItalyMap)

    

    

ItalyMap
temp_Italy = Italy_Covid19.copy()

temp_Italy['Country'] = 'Italy' # to have a single root



fig = px.treemap(temp_Italy, 

                 path=["Country", "denominazione_regione"], values="totale_casi", height=700,

                 title='Total Confirmed Cases In Italy',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()



fig = px.treemap(temp_Italy, 

                 path=["Country", "denominazione_regione"], values="totale_positivi", height=700,

                 title='Number of Currently Positive Cases In Italy',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()

fig = px.treemap(temp_Italy, 

                 path=["Country", "denominazione_regione"], values="deceduti", height=700,

                 title='Number of Deaths reported in the Italian Region',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent parent+percent entry'

fig.show()
covid19_USA = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv")

covid19_USA.head(3)
# US Data

covid_19_USA = covid_19[covid_19['Country'] == 'US']

covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Grand Princess']

covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Diamond Princess']

#covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Guam']



formatted_text('***USA Numbers -***')



# Data Glimpse

covid_19_USA.head()
USA_statewise_data = covid_19_USA.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max()



USA_statewise_data['State'] = USA_statewise_data.index

USA_statewise_data.index = np.arange(1, len(USA_statewise_data.State.unique().tolist())+1)



USA_statewise_data = USA_statewise_data[['State','Confirmed', 'Deaths', 'Recovered']]



USA_locations = covid_19_USA[['State','Lat','Long']]

USA_locations.drop_duplicates(keep='first', inplace=True)



USA_statewise_data = pd.merge(USA_statewise_data, USA_locations, on='State')



formatted_text('***Country wise numbers of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')



USA_statewise_data.head()
us_lat = 45

us_lon = -115



formatted_text('***Click on the pin to veiw details stats***')



USAMap = folium.Map(location=[us_lat, us_lon], zoom_start=4, tiles='cartodbpositron')



for lat, long, confirmed, deaths, recovered, state in zip(USA_statewise_data['Lat'],

                                                           USA_statewise_data['Long'],

                                                           USA_statewise_data['Confirmed'],

                                                           USA_statewise_data['Deaths'],

                                                           USA_statewise_data['Recovered'], 

                                                           USA_statewise_data['State']):

    

    if (deaths == 0):

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Total Confirmed:</strong> ' + str(confirmed) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(55, 83, 109)').add_to(USAMap)

    else:

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Total Confirmed:</strong> ' + str(confirmed) + '<br>'

                                 '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'

                                 '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')

                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(26, 118, 255)').add_to(USAMap)

    

    

USAMap
temp_USA = USA_statewise_data.copy()

temp_USA['Country'] = 'United States of America' # to have a single root



fig = px.treemap(temp_USA, 

                 path=["Country", "State"], values="Confirmed", height=700,

                 title='Total Confirmed Cases In USA',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()



fig = px.treemap(temp_USA, 

                 path=["Country", "State"], values="Deaths", height=700,

                 title='Total Reported Deaths In USA',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()
# Brazil Data

covid_19_Brazil = covid_19[covid_19['Country'] == 'Brazil']

covid_19_Brazil.head()
# Brazil Data

covid_Brazil = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")

#covid_Brazil

brazil_statewise_data = covid_Brazil.groupby(['state'])['cases', 'deaths'].max()

#brazil_statewise_data



brazil_statewise_data['state'] = brazil_statewise_data.index

brazil_statewise_data.index = np.arange(1, len(brazil_statewise_data.state.unique().tolist())+1)

brazil_statewise_data = brazil_statewise_data[['state', 'cases', 'deaths']]



geoBrazil = pd.read_csv("../input/brazilianstates/states.csv")



# rename state to State

geoBrazil.rename(columns={'State': 'state'}, inplace=True)

brazilian_statewise_data = pd.merge(brazil_statewise_data, geoBrazil, on='state')



#brazilian_statewise_data.drop(['uf', 'population'], axis=1, inplace=True)

brazilian_statewise_data
brazil_lat = -14.235

brazil_lon = -51.9253



formatted_text('***Click on the pin to veiw details stats***')



BrazilMap = folium.Map(location=[brazil_lat, brazil_lon], zoom_start=4, tiles='cartodbpositron')



for lat, long, confirmed, deaths, state in zip(brazilian_statewise_data['Latitude'],

                                                           brazilian_statewise_data['Longitude'],

                                                           brazilian_statewise_data['cases'],

                                                           brazilian_statewise_data['deaths'],

                                                           brazilian_statewise_data['state']):

    

    if (deaths == 0):

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Confirmed Cases:</strong> ' + str(int(confirmed)) + '<br>'

                                 '<strong>Deaths Reported:</strong> ' + str(int(deaths)) + '<br>')

                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(55, 83, 109)').add_to(BrazilMap)

    else:

        folium.Marker(location=[lat, long]

                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                               

                                 '<strong>State:</strong> ' + str(state).capitalize() + '<br>'

                                 '<strong>Confirmed Cases:</strong> ' + str(int(confirmed)) + '<br>'

                                 '<strong>Deaths Reported:</strong> ' + str(int(deaths)) + '<br>')

                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'

                    , tooltip = str(state).capitalize(), fill_color='rgb(26, 118, 255)').add_to(BrazilMap)

    

    

BrazilMap
temp_Brazil = brazil_statewise_data.copy()

temp_Brazil['Country'] = 'Brazil' # to have a single root



fig = px.treemap(temp_Brazil, 

                 path=["Country", "state"], values="cases", height=700,

                 title='Total Confirmed Cases In Brazil',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()



fig = px.treemap(temp_Brazil, 

                 path=["Country", "state"], values="deaths", height=700,

                 title='Total Reported Deaths In Brazil',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry'

fig.show()
SKor_Covid_19 = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

SKor_Covid_19.head()
SKor_Covid_19.province.unique()
SKor_Covid_19.city.unique()
### Convert 'from other city', '-' in cities to 'Others'

SKor_Covid_19['city'] = np.where(SKor_Covid_19['city'] == '-', 'Others', SKor_Covid_19['city'])

SKor_Covid_19['city'] = np.where(SKor_Covid_19['city'] == 'from other city', 'Others', SKor_Covid_19['city'])

SKor_Covid_19.city.unique()
SKor_Covid_19['latitude'] = np.where(SKor_Covid_19['latitude'] == '-', '37.00', SKor_Covid_19['latitude'])

SKor_Covid_19['longitude'] = np.where(SKor_Covid_19['longitude'] == '-', '127.30', SKor_Covid_19['longitude'])



SKor_location_data = SKor_Covid_19.drop_duplicates(subset = "city", keep = 'first', inplace = False)



SKor_location_data = SKor_location_data[['city', 'latitude', 'longitude']]
SKorea_citywise_data = pd.DataFrame(SKor_Covid_19.groupby(['city'], as_index=False)['confirmed'].sum())

SKorea_citywise_data = pd.merge(SKorea_citywise_data, SKor_location_data, on='city')



SKorea_citywise_data
SKorea_lat = 36.00

SKorea_Lon = 127.30



formatted_text('***Click on the pin to veiw details stats***')



SKoreaMap = folium.Map(location=[SKorea_lat, SKorea_Lon], zoom_start=7, tiles='cartodbpositron')



for lat, long, confirmed, city in zip(SKorea_citywise_data['latitude'],

                                                           SKorea_citywise_data['longitude'],

                                                           SKorea_citywise_data['confirmed'],

                                                           SKorea_citywise_data['city']):  

    folium.Marker(location=[lat, long]

                  , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 

                             '<strong>State:</strong> ' + str(city).capitalize() + '<br>'

                             '<strong>Total Confirmed:</strong> ' + str(confirmed) + '<br>')

                             ,icon=folium.Icon(color='blue', icon='info-sign'), color='rgb(26, 118, 255)'

                  , tooltip = str(city).capitalize(), fill_color='rgb(26, 118, 255)').add_to(SKoreaMap)

    

SKoreaMap    