# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import all required libraries
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files
import sys

#!conda install -c conda-forge geopy --yes # uncomment this line to install geopy
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this to install folium
import folium # map rendering library

#!conda install -c plotly plotly_express --yes

print('Libraries imported.')
import requests, zipfile, io
import urllib.error

mycountry_url= 'https://download.geonames.org/export/zip/US.zip'
myfile= 'US.txt'
myloc= 'New York'

# Site containing zip codes(postal codes) of every country in the world along wiht lat/long codes
try:
    r = requests.get(mycountry_url)
except urllib.error.URLError:
    sys.exit("Cannot access geonames file from: ",my_country_url)
    
z = zipfile.ZipFile(io.BytesIO(r.content))

# Reading Zip File into a dataframe
df_US= pd.read_table(z.open(myfile), header= None)
z.close()

# The dataframe contains all the cities in the US. Load only the NewYork city Zip codes into a new dataframe
# Drop columns that are not required
df_US.drop(columns=[0, 4, 6, 7, 8, 11], axis=0, inplace=True)
df_US.rename(columns = {1:'Postal Code', 2: 'Locale', 3:'State', 5: 'Area', 9:'Latitude', 10:'Longitude'}, inplace = True)

df_ny_data= pd.DataFrame(columns=['Postal Code','Locale', 'State', 'Area', 'Latitude', 'Longitude'])


for index in range(len(df_US)):
    if((df_US['State'].iloc[index]==  myloc) & (df_US['Postal Code'].iloc[index] <= 10499)):        
        df_ny_data= df_ny_data.append(pd.Series([df_US['Postal Code'].iloc[index],
                                                df_US['Locale'].iloc[index],
                                                df_US['State'].iloc[index],
                                                df_US['Area'].iloc[index],
                                                df_US['Latitude'].iloc[index],
                                                df_US['Longitude'].iloc[index]],
                                                index=df_ny_data.columns), ignore_index= True)

print('NYC zip codes (10000- 10499) and lat/long data downloaded from geonames!')
df_ny_data
df_ny_data.shape
# This is step #3 where we find the lat and long of NY City using geocode
address = 'New York,NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York,NY are {}, {}.'.format(latitude, longitude))
from folium import plugins

# create map of New York using latitude and longitude values
map_myCity = folium.Map(location=[latitude, longitude], zoom_start=10)

# instantiate a mark cluster object for the incidents in the dataframe
my_regions = plugins.MarkerCluster().add_to(map_myCity)

# add markers to map
for lat, lng, postal, locale in zip(df_ny_data['Latitude'], df_ny_data['Longitude'], df_ny_data['Postal Code'], df_ny_data['Locale']):
    label = '{}: {}, {}'.format("NY Zip", postal, locale)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(my_regions)  
    
map_myCity
CLIENT_ID = 'OM1SXYU0TXVYK115IQ0RQ4RSVCA4IXPME03N54EVLCCDFLG5' # your Foursquare ID
CLIENT_SECRET = 'I5ZW4EZLOESCLF0OPX0O3TFZMS0JW4R3AQW5BBW0S5YZZOTQ' # your Foursquare Secret
VERSION = '20200418' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
# Another option is to get a hospital json file for USA only.. 
# https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Hospitals_1/FeatureServer/0/query?where=1%3D1&outFields=ID,NAME,ADDRESS,CITY,STATE,ZIP,TYPE,STATUS,POPULATION,SOURCEDATE,OBJECTID,TELEPHONE&outSR=4326&f=json

# Code below is for foursquare based seach of hospitals around a given zip code/postal code... foursquare can work any where is 
LIMIT = 20 # limit of number of hospitals returned by Foursquare API for each neighborhood to 10

categoryID= '4bf58dd8d48988d196941735' # category for hospitals
df_hosp= pd.DataFrame()

# Copy relevant entries from the dictionary into a dataframe
for lat, lng, postal in zip(df_ny_data['Latitude'], df_ny_data['Longitude'], df_ny_data['Postal Code']):
    # create URL
    foursquare_url = 'https://api.foursquare.com/v2/venues/search?ll={},{}&categoryId={}&client_id={}&client_secret={}&limit={}&v={}'.format(
    lat, 
    lng,
    categoryID,
    CLIENT_ID, 
    CLIENT_SECRET, 
    LIMIT,
    VERSION)
 
    t_hosp= requests.get(foursquare_url).json()
    hospitals= t_hosp['response']['venues']
    df= json_normalize(hospitals)
           
    # Remove the unncessary columns
    # for each list of hospitals build a master dataframe
    df.drop(['hasPerk', 'categories','id', 'location.cc', 'location.crossStreet','location.address', 'location.city', 'location.country', 'location.formattedAddress', 'location.labeledLatLngs', 'location.state', 'referralId'], axis=1, inplace=True)
    # Create a new column that has the first part of the postal code
    new= df['location.postalCode'].str.split(" ",n= 1, expand=True)
    df["postal Prefix"]= new[0]
    df["Source Lat"]= lat
    df["Source Long"]= lng
    df["Source Postal Code"]= postal
    # Append this new data frame to the last one
    df_hosp= df_hosp.append(df, ignore_index= True)
# List the unique hospital names
hospital_names= df_hosp['name'].value_counts()
hospital_names
# Extend the df_cn_data with the list of hospitals found for each Postal code/Borough
df_ny_hospitals= pd.DataFrame(columns=['Postal Code','Locale', 'Area', 'Latitude', 'Longitude','Nearest Hospital', 'Nearest Distance'])

# Initialize temp storage
shortest_distance=[1000000]*len(df_ny_data)
nearest_hosp= [' ']*len(df_ny_data)

for index in range(len(df_ny_data)):
    shortest_distance[index]= 1000000
    nearest_hosp[index]= " "
    for index2 in range(len(df_hosp)):
        if (df_ny_data['Postal Code'].iloc[index]==  df_hosp['Source Postal Code'].iloc[index2]):
            if(df_hosp['location.distance'].iloc[index2] < shortest_distance[index]):
                shortest_distance[index]= df_hosp['location.distance'].iloc[index2]
                nearest_hosp[index]= df_hosp['name'].iloc[index2]
     # Write the details of the 
    df_ny_hospitals= df_ny_hospitals.append(pd.Series([df_ny_data['Postal Code'].iloc[index],
                                                         df_ny_data['Locale'].iloc[index],
                                                         df_ny_data['Area'].iloc[index],
                                                         df_ny_data['Latitude'].iloc[index],
                                                         df_ny_data['Longitude'].iloc[index],
                                                         nearest_hosp[index],
                                                         shortest_distance[index]],
                                                         index=df_ny_hospitals.columns),
                                                    ignore_index=True)
# Print the dataframe containing the master mappings
df_ny_hospitals
# Generate data for COVID-19 cases per neighborhood, read from excel
# Data is based on https://raw.githubusercontent.com/datasets/covid-19/master/data/key-countries-pivoted.csv using US growth rates
# The US growth rates for the prior two weeks will be randomly distributed across the borough areas
# Generate data for hospital load factors, read from excel
# Based on the number of COVID-19 cases and hospital load factors, 
# the preferred hospital will be modified to the next closest hospital
# Use gitbuhub API to get data from a given user

import urllib.error

mygitname= 'muraliinsd'
mygittoken= 'e06c0c22f9a028992aebcd0fb94066551f1db23d'
github_api = "https://api.github.com"
gh_session = requests.Session()
gh_session.auth = (mygitname, mygittoken)

url = github_api + '/repos/nychealth/coronavirus-data/commits'
commits = gh_session.get(url = url)
commits_json= commits.json()

df_json= json_normalize(commits_json, 'parents',['commit'])

df_nycovid19= pd.DataFrame()
for num in range(len(df_json)):
    df_nyc_sha= df_json['sha'].iloc[num]
    nychealth_url = 'https://raw.githubusercontent.com/nychealth/coronavirus-data/{}/tests-by-zcta.csv'.format(df_nyc_sha)
    df_nyc_date= df_json['commit'].iloc[num]
    newday= df_nyc_date['committer']['date'].split('T',1)
    try:
        df_c19_thisday= pd.read_csv(nychealth_url, names=['Postal code','Covid-19 positive', 'Total cases', 'Cum. Perc'], skiprows=[0,1])
    except urllib.error.URLError:
        continue
        
    df_c19_thisday= df_c19_thisday.astype({"Postal code": 'int64'})
    for num2 in range(len(df_c19_thisday)):
        if(df_c19_thisday['Postal code'].iloc[num2] <= 10499):   
            df_nycovid19= df_nycovid19.append(pd.Series([df_c19_thisday['Postal code'].iloc[num2],
                                                     df_c19_thisday['Covid-19 positive'].iloc[num2],
                                                     df_c19_thisday['Total cases'].iloc[num2],
                                                     df_c19_thisday['Cum. Perc'].iloc[num2],
                                                     newday[0]],index=['Postal code','Covid-19 positive','Total cases','Cum. Perc','Date']), ignore_index=True)
            
df_nycovid19= df_nycovid19[['Date', 'Postal code', 'Covid-19 positive','Total cases','Cum. Perc']]
df_nycovid19= df_nycovid19.astype({"Postal code": 'int64'})
df_nycovid19
# Let's use facebook Prophet for forecsting simple COVID-projections and load factors
from fbprophet import Prophet
import matplotlib.pyplot as plt 
%matplotlib inline

# Prophet can only predict one simply notated date and value pair series. We will separate the df by zipcode
# Get unique zip codes from the list
zipslist= list(df_nycovid19['Postal code'].unique())
df_covid19_forecast= pd.DataFrame()

for zipcode in zip(zipslist):
    df_CC= pd.DataFrame()                                  # Clear the data frame each time
    for count in range(len(df_nycovid19)):
        if(zipcode== df_nycovid19['Postal code'].iloc[count]):
            df_CC= df_CC.append(pd.Series([df_nycovid19['Date'].iloc[count],
                                                     df_nycovid19['Covid-19 positive'].iloc[count]]
                                                     ,index=['ds','y']), ignore_index=True)
    # Fit the Covid cases for each zip code and predict the future 10 days 
    m_CC = Prophet()
    m_CC.fit(df_CC)

    # Let's predict 10 days into the future
    future = m_CC.make_future_dataframe(periods=10)
    forecast = m_CC.predict(future)
    df_forecastzip= forecast[['ds','yhat']].tail(10)
    df_forecastzip.reset_index(inplace= True)
    df_zipcodes= pd.DataFrame()
    for count2 in range(len(df_forecastzip)):                      # Create a temp df with the same zipcodes
        df_zipcodes= df_zipcodes.append(pd.Series([zipcode], index=['zip']),ignore_index=True)
    df_forecastzip= df_forecastzip.assign(zipcode= df_zipcodes.get('zip'))  # Add a new zipcode colunm to the prediction
    df_covid19_forecast= df_covid19_forecast.append(df_forecastzip, ignore_index=True)
df_covid19_forecast
# The forecasts ( df_covid19_forecast) made by Prophet contains a 10 day lookahead. To show them on a map, a chorpleth would be idea but we need to 
# show the average increase counts rather than each day
# Also, the zip code is an series object.. need to convert it into an integer to be able to map to the json file

df_choro= pd.DataFrame()
current_zip= 0
for count in range(len(df_covid19_forecast)):
    int_zip= df_covid19_forecast["zipcode"].iloc[count]
    zip_value= int_zip[0]
    if(current_zip != zip_value):
        average_covid= 0
        for count2 in range(10):
            average_covid= average_covid+df_covid19_forecast["yhat"].iloc[count]
        # find the average over 10 counts
        average_covid= average_covid/count2
        current_zip= zip_value
        df_choro= df_choro.append(pd.Series([int(zip_value),average_covid],
                                            index=['postalCode', 'covid-19 Ten day increase forecast']),
                                            ignore_index=True)
df_choro= df_choro.astype({"postalCode":'int64', "covid-19 Ten day increase forecast":'int64'})
# Creating a chloropleth map to show the prediciton of covid-19 cases for the next 10 days
# Let's see them on a map along with zip codes shoiwng the closest hospital (as reported by foursquare)
# NOTE: Foursquare seems to be missing some close-in hospitals
import urllib
from urllib.request import urlopen
import json
import plotly
from folium.map import Marker, Popup
from folium import plugins

# We need a GeoJSON files that contains the boundaries of the countries. The json file by zip code can be downloaded from the github page
# https://github.com/OpenDataDE/State-zip-code-GeoJSON
# Clicking on ny_new_york_zip_codes_geo.min.json and clicking download leads to:
# https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json
# Better option: https://raw.githubusercontent.com/fedhere/PUI2015_EC/master/mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson

# download geojson file and store it into a dictionary
nyc_zipcodes=json.load(urlopen('https://raw.githubusercontent.com/fedhere/PUI2015_EC/master/mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson'))

keep_entry= [0]*len(nyc_zipcodes['features'])
for il in range(len(df_choro)):
    for zips in range(len(nyc_zipcodes['features'])):
        if(df_choro['postalCode'].iloc[il]== int(nyc_zipcodes['features'][zips]['properties']['postalCode'])):
            keep_entry[zips]= 1
            nyc_zipcodes['features'][zips]['properties']['postalCode']= int(nyc_zipcodes['features'][zips]['properties']['postalCode'])

# Let's see them on a map
# create map of NYC showing neighborhoods as blue circles and hospitals as blue markers.  
# Clicking on each neighborhood will show the name and the distance to the closest hospital
# When armed with COVID-19 incidences and hospital load factors, other hospital choices will be chosen
map_hospitals = folium.Map(location=[latitude, longitude], zoom_start=12)

# instantiate a mark cluster object for the incidents in the dataframe
my_regions = plugins.MarkerCluster().add_to(map_hospitals)

zipcode= 0;
for index in range(len(df_ny_hospitals)):
    postal= df_ny_hospitals['Postal Code'].iloc[index]
    if(zipcode != postal):
        lat= df_ny_hospitals['Latitude'].iloc[index]
        lng= df_ny_hospitals['Longitude'].iloc[index]
        area=df_ny_hospitals['Locale'].iloc[index]
        nearest_hosp= df_ny_hospitals['Nearest Hospital'].iloc[index]
        nearest_distance=df_ny_hospitals['Nearest Distance'].iloc[index]
        label = '{} {} {}, {} {} {} {} {}'.format("For Zip code: ", postal, area, "Hospital-->", nearest_hosp, "is", nearest_distance, "m away")
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7, parse_html=False
            ).add_to(my_regions)
        zipcode= postal

map_hospitals.choropleth(
    geo_data= nyc_zipcodes,
    data= df_choro,
    columns=['postalCode', 'covid-19 Ten day increase forecast'],
    key_on= 'properties.postalCode',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='covid-19 10-day forecast by NYC Zipcode')

# Let's now show the hosptials as markers from the df_hosp dataframe. Note that there are repeated names in df_hosp that needs
# to be tidied
new_df= df_hosp[['name', 'location.lat','location.lng']]
new_df= new_df.drop_duplicates(subset=['name']).reset_index()

for index in range(len(new_df)):
    hosp_name= new_df['name'].iloc[index]
    hosp_lat=  new_df['location.lat'].iloc[index]
    hosp_lng=  new_df['location.lng'].iloc[index]
    #show hospitals as markers
    label = '{}'.format(hosp_name)
    label = folium.Popup(label, parse_html=True)
    my_regions.add_child(folium.Marker(
                         location= [hosp_lat, hosp_lng],
                         popup=label))

# Hovering over each marker will show the zip code and the nearest hospital to the zipcode and the number of covid-19 cases
map_hospitals
