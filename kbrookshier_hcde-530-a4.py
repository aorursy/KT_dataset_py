citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

citytracker['Chicago']
increment = 17500

citytracker['Seattle'] += increment

print("There are currently", citytracker['Seattle'], "residents in Seattle.")
citytracker['Los Angeles'] = 45000

citytracker['Los Angeles']
print('Denver: %d' %citytracker['Denver'])
for key in citytracker.keys():

    print(key)
for key in citytracker.keys():

    print(key, ': %d' %citytracker[key])
def is_city_in_tracker(city):

    if city in citytracker:

        print(city, ': %d' %citytracker[city])

    else:

        print("Sorry, that is not in the City Tracker.")



print(is_city_in_tracker("New York"))

print(is_city_in_tracker("Atlanta"))
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for city in potentialcities:

    if city in citytracker:

        print(city, ': %d' %citytracker[city])

    else:

        print(0)
for city in citytracker:

    print("%s,%s" %(city,citytracker[city]))
import os

import csv



### Add your code here

with open('popreport.csv', 'w') as file:

    file.write('city,pop')

    for city in citytracker.keys():

        file.write("%s,%s" %(city,citytracker[city]))



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Los Angeles'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



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



# lat and lon below are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])
