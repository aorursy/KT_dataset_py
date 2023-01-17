#Importing all the libraries required.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium
# Reading data from csv to pandas dataframe.

cor = pd.read_csv("../input/Corona virus.csv")

cor.head()
#Looking at the certain aspects of the data

cor.info()
cor = cor.drop(labels='Sno',axis=1)

cor['Last Update'] = cor['Last Update'].apply(pd.to_datetime)

cor.head()
# Total countries affected along with their names

total = cor['Country'].unique().tolist()

print(total)



print("Total Countries affected by corona virus",len(total))
# Grouping the data on the basis of country

cases = cor.groupby(['Country']).sum()

cases = cases[cases['Confirmed']>0]

cases = cases.reset_index()

cases.head()
# Checking the figures worldwide

conf_case = cor['Confirmed'].sum()

deaths = cor['Deaths'].sum()

recovered = cor['Recovered'].sum()

sur = conf_case - (deaths + recovered)

print("Globally Confirmed Cases: ",conf_case)

print("Globally Death Cases: ",deaths)

print("Globally Recovered Cases: ",recovered)

print("Globally Under Survilance Cases: ",sur)


world_data = pd.DataFrame({

   'name':list(cases['Country']),

    'lat':[-25.27,12.57,56.13,39.91,61.92,46.23,51.17,22.32,20.59,41.87,36.2,22.2,35.86,4.21,28.39,12.87,1.35,35.91,7.87,23.7,15.87,37.09,23.42,14.06,],

   'lon':[133.78,104.99,-106.35,116.36,25.75,2.21,10.45,114.17,78.96,12.56,138.25,113.54,104.19,101.98,84.12,121.77,103.82,127.77,80.77,120.96,100.99,-95.71,53.84,108.28],

   'Confirmed':list(cases['Confirmed']),

})



# Creating the map and Displaying it

world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Confirmed'], world_data['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
cases1 = cases[cases['Deaths']>0]

world_data1 = pd.DataFrame({

    'name':list(cases1['Country']),

    'lat':[35.86,],

    'lon':[113.54,],

    'Deaths':list(cases1['Deaths']),

})



world_map1 = folium.Map(location=[10, -20], zoom_start=1.0,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data1['lat'], world_data1['lon'], world_data1['Deaths'], world_data1['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Death Cases</strong>: ' + str(value) + '<br>'),

                        color='Red',

                        

                        fill_color='Red',

                        fill_opacity=0.9).add_to(world_map1)

world_map1

cases2 = cases[cases['Recovered']>0]

world_data2 = pd.DataFrame({

    'name':list(cases2['Country']),

    'lat':[-25.27,36.2,35.86,15.87],

    'lon':[133.78,138.25,113.54,100.99],

    'Recovered':list(cases2['Recovered']),

})



world_map2 = folium.Map(location=[10, -20], zoom_start=1.0,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data2['lat'], world_data2['lon'], world_data2['Recovered'], world_data2['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Recovered Cases</strong>: ' + str(value) + '<br>'),

                        color='Red',

                        

                        fill_color='Red',

                        fill_opacity=0.9).add_to(world_map2)

world_map2
# Let's look at different Province/State and countries.

df = cor.groupby(['Country','Province/State']).sum()

df = df[df['Confirmed'] > 0]

df

china_df1 = cor[cor['Country']=='Mainland China']

china_df2 = cor[cor['Country']=='China']

china_df = pd.concat([china_df1, china_df2])



china_df = china_df.drop(columns=['Country','Last Update'],axis=1)



china_df = china_df.groupby('Province/State').sum()

china_df = china_df.reset_index()



china_df = china_df.drop([13], axis=0)

china_df = china_df.reset_index(drop=True)

china_df = china_df[china_df['Confirmed']>0]

china_df.head()

# Looking at the top 5 state in terms of death.

death_max = china_df.groupby('Province/State')['Deaths'].sum().sort_values(ascending = False)[:5]

death_max
# Looking at the top 5 state in terms of recovery.

recovered_max = china_df.groupby('Province/State')['Recovered'].sum().sort_values(ascending=False)[:5]

recovered_max
p , l = plt.subplots(figsize=(12,8)) 



sns.set_color_codes('pastel')

sns.barplot(x='Confirmed', y='Province/State', data=china_df, label='Confirmed', color='r')



sns.set_color_codes('dark')

sns.barplot(x='Deaths', y='Province/State', data=china_df, label='Deaths', color='r')





sns.set_color_codes('dark')

sns.barplot(x='Recovered', y='Province/State', data=china_df, label='Recovered', color='g')



l.legend(ncol=2, loc="lower right", frameon=True)

l.set(xlim=(0, 40), ylabel="", xlabel="Stats")

sns.despine(left=True, bottom=True)