citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker['Chicago'])
#citytracker.update( {'Seattle' : 724725 + 17500})
#print(citytracker['Seattle'])

citytracker['Seattle'] += 17500
print(citytracker['Seattle'])

citytracker.update( {'Los Angeles' : 45000})
print(citytracker['Los Angeles'])
x = citytracker['Denver']
print('Denver:' + str(x))
for x in citytracker.keys():
    print(x)
for x in citytracker.keys(): #iterate over every line of the dictionary
    print(x, ':', citytracker[x]) #print the key, and the value for every key
if 'New York' in citytracker:
    print('New York', ':', citytracker['New York'])
else:
    print('Sorry, that is not in the City Tracker')
    
if 'Atlanta' in citytracker:
    print('Atlanta', ':', citytracker['Atlanta'])
else: 
    print('Sorry, that is not in the City Tracker')
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities:
    if city in citytracker:
        print(city)
    else:
        print('0')
for x in citytracker.keys(): #iterate over every line of the dictionary
    s = "{0},{1}".format(x, citytracker[x]) #format the dictionary so that keys and values are separated by a comma
    print(s) #print the dictionary in this format
import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here
import csv

w = csv.writer(open("popreport.csv", "w"))
for city, pop in citytracker.items(): #items (pairs) not keys (value)!
    w.writerow([city, pop])

import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenCage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("OpenWeather") #replace "openweathermap" with the key name you created!

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(secret_value_0)
query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary
results = geocoder.geocode(query)
lat = str(results[0]['geometry']['lat'])
lng = str(results[0]['geometry']['lng'])
print (f"{query} is located at:")
print (f"Lat: {lat}, Lon: {lng}")


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
### Test with Atlanta

geocoder = OpenCageGeocode(secret_value_0)
query = 'Atlanta' # replace this city with cities from the names in your citytracker dictionary
results = geocoder.geocode(query)
lat = str(results[0]['geometry']['lat'])
lng = str(results[0]['geometry']['lng'])
print (f"{query} is located at:")
print (f"Lat: {lat}, Lon: {lng}")


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

def getForecast(city= 'Atlanta'):
    key = secret_value_1
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)

data = json.load(getForecast())
print(data)
current_time = datetime.datetime.now() 

print(f"The current weather in Atlanta is: {data['weather'][0]['description']}")
print("Retrieved at: %s" %current_time)
### Test with every city in the city tracker.

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

for x in citytracker.keys(): #iterate over every line of the dictionary
    temp_name = x #create a variable to hold the name of the new city each time it goes through the loop
    geocoder = OpenCageGeocode(secret_value_0)
    query = temp_name 
    results = geocoder.geocode(query)
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
    print (f"{query} is located at:")
    print (f"Lat: {lat}, Lon: {lng}")
    
    def getForecast(city= temp_name):
        key = secret_value_1
        url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
        print(url)
        return safeGet(url)
    
    data = json.load(getForecast())
    print(data)
    current_time = datetime.datetime.now() 
    print("The current weather in", temp_name, f"is {data['weather'][0]['description']}")
    print("Retrieved at: %s" %current_time)

