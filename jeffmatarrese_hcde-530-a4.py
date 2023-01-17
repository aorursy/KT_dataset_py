citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print("Chicago currently has {} residents".format(citytracker["Chicago"])) # .formatting ea value into string
citytracker["Seattle"] += 17500 # adding to Sea pop

print("Seattle now has {} residents".format(citytracker["Seattle"])) 
citytracker["Los Angeles"] = 45000 # assigning new key and value

print("Los Angeles has {} residents".format(citytracker["Los Angeles"]))
print("Denver: {}".format(citytracker["Denver"]))

# Alt method per the ask: 

# print("Denver: " + str(citytracker["Denver"]))
for city in citytracker.keys(): 

    print(city)
for city in citytracker.keys():

    print("{}: {}".format(city, citytracker[city]))
def memb_check(city): # new method to 

    if city in citytracker.keys():

        print("{}: {}".format(city, citytracker[city]))

    else:

        print("Sorry, that is not in the City Tracker")

        

memb_check("New York")

memb_check("Atlanta")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



default = 0

for city in potentialcities:

    if city in citytracker:

        print("{}: {}".format(city, citytracker[city]))

    else:

        print("{}: {}".format(city, default))
for city in citytracker.keys():

    print("{},{}".format(city, citytracker[city]))
import os



### Add your code here

import csv

fields = ["city", "pop"]

with open("popreport.csv", 'w', newline = "") as csvfile:

    writer = csv.writer(csvfile, dialect='excel', delimiter=',')

    writer.writerow(fields) 

    for city in citytracker.keys():

        rowdict = [str(city),str(citytracker[city])]

        writer.writerow(rowdict)



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darksky")

secret_value_1 = user_secrets.get_secret("opencage")

key1 = secret_value_1



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
# COMMENTED OUT SINCE I GET BOTH KEYS ABOVE

# This code retrieves your key from your Kaggle Secret Keys file

# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()

# secret_value_0 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key



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

def getForecast(lat,lng): # CLEARED OUT DEFAULTS

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

print(str(data['daily']['summary']))
# new function for printing the weather report

def print_weather_report():

    for city in citytracker.keys():

        # iterate through each city to return lat and long values

        query = city

        results = geocoder.geocode(query)

        lat = str(results[0]['geometry']['lat'])

        lng = str(results[0]['geometry']['lng'])



        # run getForecast method

        data = json.load(getForecast(lat,lng))

        # printing out results as a string, with a new line between each

        print("In {} it is currently {} and {} degrees.".format(city,data['currently']['summary'],data['currently']['temperature']))

        

# call the function

print_weather_report()

import json



def weather_report_toJSON():

    #blank dictionary to hold current weather

    currentWeather = {}

    for city in citytracker.keys():

        # iterate through each city to return lat and long values

        query = city

        results = geocoder.geocode(query)

        lat = str(results[0]['geometry']['lat'])

        lng = str(results[0]['geometry']['lng'])



        # run getForecast method

        data = json.load(getForecast(lat,lng))

        # write city and weather data to dictionary

        currentWeather[city] = []

        currentWeather[city].append({

            'summary': data['currently']['summary'],

            'temperature': data['currently']['temperature']

        })

    

    # now write currentWeather dictionary to json file

    with open('currentWeather.json', 'w', encoding='utf-8') as f:

        json.dump(currentWeather, f, ensure_ascii=False, indent=4)

    

weather_report_toJSON()