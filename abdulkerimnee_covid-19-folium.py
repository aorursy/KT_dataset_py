import pandas as pd
import folium 
import geopandas as gpd
import requests
data=requests.get('https://www.worldometers.info/coronavirus/')
data_corana=pd.read_html(data.text)
for data_covid in data_corana:
    print(data_covid)
data_covid=data_covid[1:212]
data_covid[['Country,Other','TotalCases','NewCases','TotalDeaths','NewDeaths']]
m=folium.Map(zoomstart=5,
            tiles='cartodbdark_matter')
m
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'
data_covid.replace('USA', "United States of America", inplace = True)
data_covid.replace('Tanzania', "United Republic of Tanzania", inplace = True)
data_covid.replace('DRC', "Democratic Republic of the Congo", inplace = True)
data_covid.replace('Congo', "Republic of the Congo", inplace = True)
data_covid.replace('Lao', "Laos", inplace = True)
data_covid.replace('Syrian Arab Republic', "Syria", inplace = True)
data_covid.replace('Serbia', "Republic of Serbia", inplace = True)
data_covid.replace('Czechia', "Czech Republic", inplace = True)
data_covid.replace('UAE', "United Arab Emirates", inplace = True)
data_covid.replace('UK', "United Kingdom", inplace = True)
data_covid.replace('S. Korea', "South Korea", inplace = True)
data_covid.replace('North Macedonia', "Macedonia", inplace = True)
data_covid.replace('Guinea-Bissau', "Guinea Bissau", inplace = True)
data_covid.replace('Bahamas', "The Bahamas", inplace = True)
data_covid.replace('Eswatini', "Swaziland", inplace = True)
data_covid.replace('Timor-Leste', "East Timor", inplace = True)
country=pd.read_csv("../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv")
country.head()
country.replace('United States', "United States of America", inplace = True)
country=country.dropna(subset=['latitude'])
country=country.dropna(subset=['longitude'])
data_covid.rename(columns={'Country,Other':'country'},inplace=True)
combine=data_covid.merge(country,on='country')
combine.head()
for country,lat,lon,totcases,newcases,totdeaths,newdeaths in zip(combine['country'],combine['latitude'],combine['longitude'],combine['TotalCases'],combine['NewCases'],combine['TotalDeaths'],combine['NewDeaths']):
    folium.CircleMarker(
        radius=totcases*0.0001,
        color='crimson',
        fill='crimson',
        location=[lat,lon],
        popup=('Country:'+str(country),
               'Total Cases:'+str(totcases),
               'New Cases:'+str(newcases),
               'Total Deaths:'+str(totdeaths),
               'New Deaths:'+str(newdeaths)),
        tooltip=('<li><bold>Country:'+str(country),
                 '<li><bold>Total Cases:'+str(totcases),
                 '<li><bold>New Cases:'+str(newcases),
                 '<li><bold>Total Deaths:'+str(totdeaths),
                 '<li><bold>New Deaths:'+str(newdeaths))
).add_to(m)
m
