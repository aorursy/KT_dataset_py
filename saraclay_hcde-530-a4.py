citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



# Printing the number of residents in Chicago

citytracker['Chicago']
# print(type(citytracker['Seattle']))

# this is an int so I can just add 17500 to the value

citytracker['Seattle']+17500
citytracker['Los Angeles']=45000

citytracker['Los Angeles']

# I printed the array just to make 100% that this entry was added

# citytracker
# This is what I thought of when I heard concatenate. Is this ok?

print("Denver: %r" %citytracker['Denver'])
for i in citytracker:

    print (i)
for i in citytracker:

    print(i + ": " + str(citytracker[i]))
# The below code loops through each entry to see if it has New York or not. It's ok, but there must be a more efficient way of doing this.

# for i in citytracker:

#    if "New York" in i:

#        print("New York: " + str(citytracker["New York"]))

#    else:

#        print("Sorry that is not in the City Tracker.")



# I wanted to get fancy and use a default value!

def is_there_city(i="New York"):

        if (i in citytracker):

            print(i + ": " + str(citytracker[i]))

        else:

            print("Sorry, that is not in the city tracker.")

        

        

is_there_city()

is_there_city("Atlanta")

        
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
# We have a list that we want to check against a dictionary

# Writing this out in the previous exercise helped to solve this one



def potential_cities(i="Cleveland"):

     for i in potentialcities:

            if (i in citytracker):

                print(i + ": " (citytracker[i]))

            else:

                print(0)

    

potential_cities()
for i in citytracker:

    print(i + "," + str(citytracker[i]))
import os

import json



### Add your code here

### My only nit with this is that the first and last items start and end with a bracket, respectively...

print("city,pop")

for i in citytracker:

    print(i + "," + str(citytracker[i]))



with open('popreport.csv', 'w') as outfile:

    json.dump(citytracker, outfile)



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



for i in citytracker:

# query = "Los Angeles"  # replace this city with cities from the names in your citytracker dictionary

    query = i

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (i + " - Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSky") # make sure this matches the Label of your key



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

# def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



# looping through all of the information from dark sky

for i in citytracker:

    results = geocoder.geocode(i)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    print("=========" + "Retrieved at: %s" %current_time + "=========")

    print("The weather for " + i + " is...")

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])

import os

import json

import urllib.error, urllib.parse, urllib.request, json, datetime



# Error handling

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



# Calling Dark Sky

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



for i in citytracker:

    results = geocoder.geocode(i)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now()

    # Making the current weather a variable to make things easier

    weather_now = data['currently']['summary']

    # Switching out the former city tracker value with a *new* value

    citytracker[i]=weather_now



with open('weatherreport.csv', 'w') as outfile:

    json.dump(citytracker, outfile)



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))