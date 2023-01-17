citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Chicago']
citytracker['Seattle'] = citytracker['Seattle'] +17500

print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000

print(citytracker)
print('Denver:' + str(citytracker['Denver']))
for city in citytracker:

    print (city)

    
for city in citytracker:

    print (city + " : " + str(citytracker[city]))

          

if 'New York' in citytracker:

    print('New York' + ':' + citytracker['New York'])

else:

    print("Sorry, that is not in the City Tracker.")

    

if 'Atlanta' in citytracker:

    print('Atlanta' + ' : ' + str(citytracker['Atlanta']))

else:

    print("Sorry, that is not in the City Tracker.")

    
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for city in potentialcities:

    if city in citytracker:

        print (city + " : " + str(citytracker[city]))

    else:

        print(0)







    
for city in citytracker:

    print (city + "," + str(citytracker[city]))

    

print(citytracker[city])
import os

import csv

    

with open("popreport.csv", "w", newline = '') as f:

    writer = csv.writer(f)

    writer.writerow(['city','pop'])

    

    for key, value in citytracker.items():

        writer.writerow([key, value])



        

### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Atlanta'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage") # make sure this matches the Label of your key

key1 = secret_value_0

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darksky") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



new_dict ={}



for city in citytracker: 

    query = city  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])



    def safeGet(url):

        try:

            return urllib.request.urlopen(url)

        except urllib2.error.URLError as e:

            if hasattr(e,"code"):

                print("The server couldn't fulfill the request.")

                print("Error code: ", e.code)

            elif hasattr(e,'reason'):

                print("We failed to reach a server")

                print("Reason: ", e.reason)

            return None



    def getForecast(lat,lng): 

        # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)



    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now()

    current_sum = data['currently']['summary']

    temp = str(data['currently']['temperature'])

    min_sum = data['minutely']['summary']

    print(city)

    print("Retrieved at: %s" %current_time)

    print(current_sum)

    print("Temperature: " + temp)

    print(min_sum)



    new_dict[city]={

        'city': city,

        'lat': lat,

        'long': lng,

        'time': current_time,

        'current': current_sum,

        'temperature': temp,

        'min summary': min_sum,

       }





print(new_dict)
import json

json_file = json.dumps(new_dict, indent=4, sort_keys=True, default=str)

print(json_file)
