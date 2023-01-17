# installation

! pip install -q calmap

! pip install -q requests

! pip install -q geopy
# import

# essential libraries

import json

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import calmap

import folium



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# date and time

import time

from datetime import datetime



# requests for getting data file from Google Sheet link

import requests



# geopy + geocoder = Nominatim for searching location using OpenStreetMap data.

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="Vietnam SARS-CoV-2 / COVID-19 Infographics", timeout=3)
# define function to get latitude, longitude

def latlongGet (addressStr):

    #addressStr = "Cu Chi, Ho Chi Minh, Vietnam"

    #addressStr = "Cu Chi District, Ho Chi Minh, Vietnam"



    location = geolocator.geocode(addressStr)

    if location is None:

        print("Cannot find address", addressStr)

        #return NaN, NaN;

    else:

        #print(location.address)

        print((location.latitude, location.longitude))

        return location.latitude, location.longitude;
# get latest Vietnam SARS-CoV-2 | COVID-19 data

import io

from io import BytesIO



# get data from shared Google Sheet

response = requests.get('https://docs.google.com/spreadsheet/ccc?key=1i2ox2Ii-SCt1qiv3I3UxO37LSVWFdq7EJR1_ETaxx6M&output=csv')

assert response.status_code == 200, 'Wrong status code'

data = response.content



# import data to dataframe

df = pd.read_csv(BytesIO(data)) #unprocessed data



# print few rows

print(df.head(5))
# process data frame



# select cols of interest

dfi = df[['Case', 'Current Location', 'Confirmed', 'Recovered']]



# create an unattached column with an index

dfState = dfi[['Current Location']]

dfState.columns = ['State']

#print(dfState)



# extract state / province name from dfState

nrow,_ = dfState.shape    

for i in range(nrow):

    addr = dfState.iloc[i,0]

    if (str(addr) == 'nan'):

        print('index = ', i, ' addr = ', addr, ' -> no address')

    else:

        s = addr.split(',')    #delimiter = ','

        state = s[len(s) - 2]  #get province / state

        state = state.strip()

        dfState.at[i,'State'] = state

        #print('state = ', dfState.iloc[i,1])

print(dfState)



# attach dfState to dfi

dfiNew = pd.concat([dfi, dfState], axis=1)

#print(dfiNew.tail(10))



# select only confirmed cases & not recovered yet

dfi = dfiNew[(dfiNew['Confirmed']==1) & (dfiNew['Recovered'].isnull())]

#print(dfi)

print(dfi.head(10))

print(dfi.tail(10))
'''

# test latlongGet function + searchable locations stored in data file

addressStr = "Cu Chi District, Ho Chi Minh, Vietnam"

addressStr = "Dong Anh, Hanoi, Vietnam"

#addressStr = "Binh Thuan Province, Vietnam"

#addressStr = "Ninh Binh Province, Vietnam"

#addressStr = "District 10, Ho Chi Minh, Vietnam"

#addressStr = "Lao Cai, Vietnam"

#addressStr = "Hanoi, Vietnam"

#addressStr = "Da Nang, Vietnam"

#addressStr = "Hoi An, Quang Nam, Vietnam"

#addressStr = "Pasteur Hospital, Ho Chi Minh, Vietnam"

#addressStr = "Hue, Vietnam"

#addressStr = "Cao Xanh, Ha Long, Quang Ninh Province, Vietnam"

#addressStr = "District 12, Ho Chi Minh, Vietnam"



try:

    lat, long = latlongGet(addressStr)

    print(lat, long)

except:

    print('')

'''
#get no. of rows in dfi

nrow,_ = dfi.shape

#print(nrow)

#print(dfi.iloc[1,1])



'''

# test getting lat, long from dataframe of interest



for i in range(nrow):

    time.sleep(1) #delay 1s to avoid #except OSError as err: # timeout error

    addr = dfi.iloc[i,1]

    if (str(addr) == 'nan'):

        print('index = ', i, ' addr = ', addr, ' -> no address')

    else:

        lat, long = latlongGet(addr)

        print('index = ', i, ' addr = ', addr, ' -> ', lat, long)

'''
# create map

# country center - position country map in the middle

centerLat = 16.4637 #Hue city Lat Long

centerLong = 107.5909



# display country map

m = folium.Map(location=[centerLat, centerLong], tiles='cartodbpositron',

               min_zoom=1, max_zoom=10, zoom_start=6)



# add SARS-CoV-2 | COVID-19 areas

#nrow = 10 #fix value for testing purpose



dfi["Lat"] = np.nan

dfi["Long"] = np.nan



for i in range(0, nrow):

    time.sleep(1) #delay 1s to avoid #except OSError as err: # timeout error

    addr = dfi.iloc[i,1]

    print(i,"-", addr)

    if (str(addr) == 'nan'):

        print('no address')

    else:

        #search if address appear once, lat long are available already, then no need to retrieve lat long again

        alreadyExist = 0

        foundPos = 0

        for j in range(0, i-1):

            addr2 = dfi.iloc[j,1]

            if (str(addr) == str(addr2)):

                alreadyExist = 1

                foundPos = j

                break

        if (alreadyExist):

            lat = dfi.iloc[foundPos,5]  #col Lat

            long = dfi.iloc[foundPos,6] #col Long

        else:

            try:

                lat, long = latlongGet(addr)

                print(lat, long)

            except:

                print('')

        

            if(lat):

                dfi.iloc[i,5] = lat         #col Lat

            if(long):

                dfi.iloc[i,6] = long        #col Long

        #print(lat, long)

    

    state = dfi.iloc[i]['State'].strip()

    print(state)

    noCaseInState = dfi.State.str.count(state).sum()

    print(noCaseInState)

    

    folium.Circle(

        location=[lat, long],

        color='crimson', 

        tooltip =   '<li><bold>Province : '+str(dfi.iloc[i]['State'])+

                    '<li><bold>Confirmed : '+str(noCaseInState),

                    #'<li><bold>Recovered : '+str(dfi.iloc[i]['Recovered']),

                    #'<li><bold>Deaths : '+str(dfi.iloc[i]['Deaths'])+

        radius=int(noCaseInState)*10).add_to(m)

'''

#testing to find total cases for each state

print(dfi.groupby(["Confirmed", "State"]).size())

print(dfi.groupby(["Confirmed", "State"]).size()[1])

print(dfi.groupby(["Confirmed", "State"]).size()[1][1])

print(dfi.groupby(["Confirmed", "State"]).size()[1]['Hanoi'])

'''

#display map

m