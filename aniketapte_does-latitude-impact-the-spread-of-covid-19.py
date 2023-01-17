# Import Python Packages

import pandas as pd

import numpy as np

import plotly.express as px

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline 

# Load Data

coordinates = pd.read_csv('/kaggle/input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv')

country_coordinates = coordinates[['country_code','latitude','longitude','country']]

state_coordinates = coordinates[['usa_state_code','usa_state_latitude','usa_state_longitude','usa_state']]

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df['Country/Region'].replace(['Mainland China'], 'China',inplace=True)

df['Country/Region'].replace(['US'], 'United States',inplace=True)

df['Country'] = df['Country/Region']

df = df[df.ObservationDate==np.max(df.ObservationDate)]

todays_date = '3/31/2020' # Update this line every time that you rerun the notebook



# Mortality rate for every country in the dataset

df_deaths = pd.DataFrame(df.groupby('Country')['Deaths'].sum())

df_confirmed = pd.DataFrame(df.groupby('Country')['Confirmed'].sum())

df_confirmed['Deaths'] = df_deaths['Deaths']

df_global = df_confirmed

df_global['Mortality Rate'] = np.round((df_global.Deaths.values/df_global.Confirmed.values)*100,2)

df_global = df_global.reset_index()

df_global = df_global.merge(country_coordinates, left_on='Country', right_on='country')

df_global = df_global[['Country','Confirmed','Deaths','Mortality Rate','latitude','longitude','country_code']]

df_global.columns = ['Country','Confirmed','Deaths','Mortality Rate','Latitude','Longitude','Country_Code']

df_global.to_csv('/kaggle/working/global_covid19_mortality_rates.csv')



# Mortality rate for every state in the USA

df_usa = df[df['Country/Region']=='United States']

df_usa = df_usa[df_usa.ObservationDate==np.max(df_usa.ObservationDate)]

df_usa['State'] = df_usa['Province/State']

df_usa['Mortality Rate'] = np.round((df_usa.Deaths.values/df_usa.Confirmed.values)*100,2)

df_usa.sort_values('Mortality Rate', ascending= False).head(10)

df_usa = df_usa.merge(state_coordinates, left_on='State', right_on='usa_state')

df_usa['Latitude'] = df_usa['usa_state_latitude']

df_usa['Longitude'] = df_usa['usa_state_longitude']

df_usa = df_usa[['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','usa_state_code']]

df_usa.columns = ['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','USA_State_Code']

df_usa.to_csv('/kaggle/working/usa_covid19_mortality_rates.csv')
covid19= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

covid19.head()

covid19.columns

df_corona = covid19[['id','case_in_country','reporting date','location','country','gender', 'age', 'symptom_onset',

       'If_onset_approximated', 'hosp_visit_date', 'exposure_start',

       'exposure_end', 'visiting Wuhan', 'from Wuhan', 'death', 'recovered',

       'symptom', 'source', 'link']]

df_corona.count()






covidall=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

covidall=covidall[['ID', 'age', 'sex', 'city', 'province', 'country',

       'wuhan(0)_not_wuhan(1)', 'latitude', 'longitude', 'geo_resolution',

       'date_onset_symptoms', 'date_admission_hospital', 'date_confirmation',

       'symptoms', 'lives_in_Wuhan', 'travel_history_dates',

       'travel_history_location', 'reported_market_exposure',

       'additional_information', 'chronic_disease_binary', 'chronic_disease',

       'source', 'sequence_available', 'outcome', 'date_death_or_discharge',

       'notes_for_discussion', 'location', 'admin3', 'admin2', 'admin1',

       'country_new', 'admin_id', 'data_moderator_initials']]

covidallsym=covidall[['ID', 'age', 'sex', 'city','country','latitude', 'longitude', 'geo_resolution','symptoms', 'lives_in_Wuhan', 'travel_history_dates',

       'travel_history_location','additional_information','notes_for_discussion']]

covidallitly=covidallsym[covidallsym['country']=='United States']



#covidallitly = covidallitly[covidallitly['symptoms'].notnull()]

covidallitly.head(50)
df_corona_death=df_corona[df_corona.death!='0']

df_corona_death.count()

df_corona.count()
df_corona_death1=df_corona_death[df_corona_death.age<40]

df_corona_death1.head()

df_corona_death1.count()

df_corona_death1.groupby('country').size()


df_corona_death2=df_corona_death[df_corona_death.age>=40]

df_corona_death2.head()

df_corona_death2.groupby('country').size()
fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Confirmed", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,5000],

                    title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Deaths", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,50],

                    title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Mortality Rate", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,5],

                    title='Global COVID-19 Mortality Rates as of '+todays_date)

fig.show()
fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:10], 

             x="Country", 

             y="Confirmed",

             title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:10], 

             x="Country", 

             y="Deaths",

             title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:10], 

             x="Country", 

             y="Mortality Rate",

             title='Global COVID-19 Mortality Rates as of '+todays_date+' for Countries with Top 10 Most Deaths')

fig.show()
fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Confirmed", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,5000],scope="usa",

                    title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Deaths", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,100],scope="usa",

                    title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_usa, 

                    locations="USA_State_Code", 

                    color="Mortality Rate", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,5],scope="usa",

                    title='Global COVID-19 Mortality Rate as of '+todays_date)

fig.show()
fig = px.bar(df_usa.sort_values('Confirmed',ascending=False)[0:10], 

             x="State", 

             y="Confirmed",

             title='USA COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:10], 

             x="State", 

             y="Deaths",

             title='USA COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:10], 

             x="State", 

             y="Mortality Rate",

             title='USA COVID-19 Mortality Rates as of '+todays_date+' for USA States with Top 10 Most Deaths')

fig.show()
df_global2 = df_global

df_global2['Latitude'] = abs(df_global2['Latitude'])

#df_global2 = df_global2[df_global2['Country']!='China']



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

             y="Mortality Rate",

             title='Global COVID-19 Mortality Rates vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

df_global.sort_values('Mortality Rate', ascending= False).head(10)
df_usa.columns
df_usa1 = df_usa

df_usa1['Latitude'] = abs(df_usa1['Latitude'])

#df_global2 = df_global2[df_global2['Country']!='China']



fig = px.scatter(df_usa1.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Confirmed",

             title='USA COVID-19 Infections vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

df_global12 = df_global[(df_global['Latitude'] >= 25) & (df_global['Latitude'] <= 65)]

#df_global12=df_global[df_global.Latitude>30 & df_global.Latitude<55]

df_global12.head()


fig = px.choropleth(df_global12, 

                    locations="Country", 

                    color="Confirmed", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,5000],

                    title='Global COVID-19 Confirmed between lattitudes 25 and 60 as of '+todays_date)

fig.show()
fig = px.choropleth(df_global12, 

                    locations="Country", 

                    color="Deaths", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,4000],

                    title='Global COVID-19 Deaths between lattitudes 25 and 60 as of '+todays_date)

fig.show()
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



df1 = df_global12



geometry = [Point(xy) for xy in zip(df1['Longitude'], df1['Latitude'])]

gdf = GeoDataFrame(df1, geometry=geometry)   



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=30);


fig = px.scatter(df_usa.sort_values('Deaths',ascending=False), 

             x="Latitude", 

             y="Mortality Rate",

             title='USA States COVID-19 Mortality Rates vs Absolute Value of Latitude Coordinate as of '+todays_date)

fig.show()

df_usa.sort_values('Deaths', ascending= False).head(10)
df_global2.head()
chinastate_coordinates= pd.read_csv('../input/chinese-cities/china_coordinates.csv')

chinastate_coordinates.head()
#country_coordinates = coordinates[['country_code','latitude','longitude','country']]

#state_coordinates = coordinates[['China_state_code','China_state_latitude','China_state_longitude','China_state']]

df_China = df[df['Country/Region']=='China']

df_China = df_China[df_China.ObservationDate==np.max(df_China.ObservationDate)]

df_China['State'] = df_China['Province/State']

df_China['Mortality Rate'] = np.round((df_China.Deaths.values/df_China.Confirmed.values)*100,2)

df_China.sort_values('Mortality Rate', ascending= False).head(10)

df_China = df_China.merge(chinastate_coordinates, left_on='State', right_on='admin')

df_China['Latitude'] = df_China['lat']

df_China['Longitude'] = df_China['lng']

df_China['China_state'] = df_China['admin']

df_China = df_China[['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','China_state']]

df_China.columns = ['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','China_state']

df_China.to_csv('/kaggle/working/China_covid19_mortality_rates.csv')

df_China.head()
df_Italy = df[df['Country/Region']=='Italy']

#df_Italy = df_Italy[df_Italy.ObservationDate==np.max(df_Italy.ObservationDate)]

df_Italy['State'] = df_Italy['Province/State']



df_Italy.head()
from pandas import DataFrame

from pandas import Series

import matplotlib.pyplot as plt

earthquake_data = df_China

earthquake_data.shape

latitude_list = []

longitude_list= []



for row in earthquake_data.Latitude:

    latitude_list.append(row)

for row in earthquake_data.Longitude:

    longitude_list.append(row)



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

%matplotlib inline

earthquake_map = Basemap(projection='robin', lat_0=-90, lon_0=130,resolution='c', area_thresh=10000.0)

earthquake_map.drawcoastlines()

earthquake_map.drawcountries()

earthquake_map.drawmapboundary()

earthquake_map.bluemarble()

earthquake_map.drawstates()

earthquake_map.drawmeridians(np.arange(0, 360, 30))

earthquake_map.drawparallels(np.arange(-90, 90, 30))



x,y = earthquake_map(longitude_list, latitude_list)

earthquake_map.plot(x, y, 'ro', markersize=1)

plt.title("Locations in China for Covid19")

 

plt.show()
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



df = df_China



geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

gdf = GeoDataFrame(df, geometry=geometry)   



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



df = df_global12



geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

gdf = GeoDataFrame(df, geometry=geometry)   



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);