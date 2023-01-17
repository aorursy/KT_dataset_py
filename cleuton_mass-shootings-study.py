import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



rawdata = pd.read_csv('../input/us-mass-shootings-nan-coordinates-fixed/mass_shootings_dataset_coords_fixed.csv',encoding = 'ISO-8859-1', parse_dates=['Date'])
rawdata.head()
rawdata[['Date', 'Total victims']].groupby([(rawdata.Date.dt.year)])['Total victims'].sum()

rawdata[(rawdata.Date.dt.year==2017)]
coords = rawdata[['Longitude', 'Latitude']].dropna()

coords.plot(kind='scatter',x='Longitude',y='Latitude')
rawdata[(rawdata.Latitude > 60)]
rawdata[(rawdata.Latitude < 25)]
rawdata[(rawdata.Longitude > -85)].max()


from scipy.misc import imread

import matplotlib.cbook as cbook

rawdata.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,

    s=rawdata["Total victims"]*1.5, label="Victims", figsize=(20,9),

    c="Total victims", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()



'''

import urllib.request

import json

url = 'https://maps.googleapis.com/maps/api/geocode/json?address=Kalamazoo&key=SORRY-USE-YOURS'

req = urllib.request.Request(url)



r = urllib.request.urlopen(req).read()

response = json.loads(r.decode('utf-8'))



for item in response['results']:

    print("Latitude:", item['geometry']['location']['lat'], 

          "Longitude:",item['geometry']['location']['lng'] )

'''
missing=rawdata[(np.isnan(rawdata.Latitude)) | (np.isnan(rawdata.Longitude))]
'''

import urllib.parse

def get_location(address):

    #print('input address:',address)

    encoded=urllib.parse.quote_plus(address)

    print('Encoded:',encoded,'address:',address)

    url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + encoded +'&key=[USE YOUR API KEY]'

    req = urllib.request.Request(url)

    r = urllib.request.urlopen(req).read()

    response = json.loads(r.decode('utf-8'))

    results = response['results']

    print('latitude:',results[0]['geometry']['location']['lat'], 'longitude:',results[0]['geometry']['location']['lng'])

    if len(results) > 0:

        return results[0]['geometry']['location']['lat'], results[0]['geometry']['location']['lng']

    else:

        return 0,0



coords=[]

for address in missing['Location']:

    lat,long = get_location(address)

    coords.append((lat,long))

coords

'''
#tuplas_coord = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])



#rawdata.loc[(np.isnan(rawdata.Latitude)) | (np.isnan(rawdata.Longitude)), ['Latitude', 'Longitude']] = tuplas_coord[['Latitude', 'Longitude']]

#rawdata

#rawdata = pd.read_csv('mass_shootings_dataset_coords_fixed.csv',encoding = 'ISO-8859-1', parse_dates=['Date'])
rawdata.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,

    s=rawdata["Total victims"]*1.5, label="Victims", figsize=(20,9),

    c="Total victims", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()


