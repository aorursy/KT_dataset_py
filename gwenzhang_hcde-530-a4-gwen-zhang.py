citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle'] += 17500
print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000
print(citytracker['Los Angeles'])
print("Denver: " + str(citytracker["Denver"]))
for city in citytracker:
    print(city)
for city in citytracker:
    print(city + ": " + str(citytracker[city]))
#define a function that take city and citytracker dictionary as input
def cityTester(city, citytracker):
    if city in citytracker:
        print(city + ": " + str(citytracker[city]))
        return;
    print("Sorry, that is not in the City Tracker.")
    return;

cityTester("New York", citytracker)
cityTester("Atlanta", citytracker)
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
#assume print zero means print both cityname and value zero.
#which could utilize default value.
for city in potentialcities:
        print(city + ": " + str(citytracker.get(city, 0)))
for city in citytracker:
    print(city + "," + str(citytracker[city]))
import os

### Add your code here
f = open("popreport.csv", "w")

f.write("city,pop\n")
for city in citytracker:
    f.write(city + "," + str(citytracker[city]) + "\n")
f.close()


### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenCage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("WeatherMap") #replace "openweathermap" with the key name you created!

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(secret_value_0)
query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary
results = geocoder.geocode(query)
lat = str(results[0]['geometry']['lat'])
lng = str(results[0]['geometry']['lng'])
print (f"{query} is located at:")
print (f"Lat: {lat}, Lon: {lng}")
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

def getForecast(city="Seattle"):
    key = secret_value_1
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)

data = json.load(getForecast())
print(data)
current_time = datetime.datetime.now() 

print(f"The current weather in Seattle is: {data['weather'][0]['description']}")
print("Retrieved at: %s" %current_time)
#import UserSecretsClient 
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenCage")
secret_value_1 = user_secrets.get_secret("WeatherMap")

from opencage.geocoder import OpenCageGeocode

cityDetail = {}
geocoder = OpenCageGeocode(secret_value_0)
# for each city fetched all necessary info from 
# each API.
for city in citytracker:
    results = geocoder.geocode(city)
    weather = json.load(getForecast(city))
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
    weatherdetail = str(weather['weather'][0]['description'])
    current_time = datetime.datetime.now() 
    #create a string which contains all the info fetched from all APIs.
    cityDetail[city] = f"{city} is located at:\nLat: {lat}, Lon: {lng}\nThe current weather in {city} is: {weatherdetail}\nRetrieved at: {current_time}\n"
    
    
#print out all content
for city in cityDetail:
    print (cityDetail[city])
    
#write it out as json with time stamp
import json

with open('finalResult.txt', 'w') as outfile:
    json.dump(cityDetail, outfile)

