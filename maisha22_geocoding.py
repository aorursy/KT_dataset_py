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
##2_comment-in for using the original google maps api

# If you have a Google Places API key, enter it here

# api_key = 'AIzaSy___IDByT70'

# https://developers.google.com/maps/documentation/geocoding/intro



api_key = False 



#3_API handling

if api_key is False:

    api_key = 42

    serviceurl = 'http://py4e-data.dr-chuck.net/json' #proxy api

else :

    serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json' #not using this



#4_File handling

in_file = open("../input/geocoding-input-file/Prac.csv","r")



lines = in_file.readlines() 

d = {} 



#5_exporting all results to another csv

out_file = open("OutputPrac.csv","w") 

out_file.write(' ')

out_file.write('\n')



#6_loop handling json data and respond.get() method

for line in lines[1:]: 

    line = line.strip() 

    address = line #maintain homogeneity

    payload = dict() #{}

    payload['address'] = address 

    if api_key is not False: payload['key'] = api_key #needed for the original API

 

    r = requests.get(serviceurl, params=payload)

    data = r.text

   

    try:

        js = json.loads(data)

    except:

        js = None 

       

    if line not in d: #for avoiding repeatations

        try:

            lat = js['results'][0]['geometry']['location']['lat']

            lng = js['results'][0]['geometry']['location']['lng']

            d[line] = [lat, lng]

            str = '{},{},{}'.format(line, d[line][0], d[line][1])

            if 20.86382 < lat < 26.33338 and 88.15638 < lng < 92.30153: #Limiting coordinates for Bangladesh

                print(str)

            else:

                print(address + 'WRONG DATA') #for data discrepancy

        except:

            print(address + 'NOT FOUND') #data not retrievable

           

        out_file.write(str)

        out_file.write('\n')

       



in_file.close()

out_file.close()



print("FINISHED")