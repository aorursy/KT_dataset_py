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
#1_importing request to send http request and get response and json to handle json data 

import requests

import json



api_key = False

# If you have a Google Places API key, enter it here

# api_key = 'AIzaSy___IDByT70'

# https://developers.google.com/maps/documentation/geocoding/intro



if api_key is False:

    api_key = 42

    serviceurl = 'http://py4e-data.dr-chuck.net/json'

else :

    serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json'



while True:

    address = input('Enter location: ')

    if len(address) < 1: break



    payload = dict()

    payload['address'] = address

    if api_key is not False: payload['key'] = api_key



    r = requests.get(serviceurl, params=payload)

#     print('Retrieved', r.url)

    data = r.text

#     print('Retrieved', len(data), 'characters')



    try:

        js = json.loads(data)

    except:

        js = None



    if not js or 'status' not in js or js['status'] != 'OK':

        print('==== Failure To Retrieve ====') #data not retrievable

        print(data)

        continue



#     print(json.dumps(js, indent=4))

    

    lat = js['results'][0]['geometry']['location']['lat']

    lng = js['results'][0]['geometry']['location']['lng']

    if 20.86382 < lat < 26.33338 and 88.15638 < lng < 92.30153: #Limiting coordinates for Bangladesh

        print('lat', lat, 'lng', lng)

    else:

        print(address + 'WRONG DATA') #for data discrepancy

#     location = js['results'][0]['formatted_address']

#     print(location)




