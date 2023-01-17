citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#access and print the key named 'Chicago' in the citytracker dictionary.
print (citytracker['Chicago'])
citytracker['Seattle'] = citytracker['Seattle'] + 17500    #update population by adding 17500 to current population value
print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000   #add Los Angeles to citytracker dictionary
print(citytracker['Los Angeles'])
x = citytracker['Denver']        # define x as population of Denver in citytracker dictionary
print('Denver: ' + str(x))
for city in citytracker.keys():         #we only loop through citytracker keys and print the city for each.
    print(city)
for city in citytracker.keys():         #looping through the citytracker keys 
    count = citytracker[city]          #we store the population values of each city so we can print them later
    print(city + ': ' + str(count))
    
    # this tests if the city New York is in the citytracker dictionary
    if 'New York' in citytracker.keys():
        print('New York: ' + str(citytracker['New York']))
    #else print that it is not in city tracker dictionary
    else:       
        print('Sorry, that is not in the City Tracker')
        
    #same idea here,
    if 'Atlanta' in citytracker.keys():
        print('Atlanta: ' + str(citytracker['Atlanta']))
    else:
        print('Sorry, that is not in the City Tracker')

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

# we first loop through potentialcities dictionary
for pcity in potentialcities:
    
    # if the city in potential city is also in citytracker dictionary, then print the name and population.
    if pcity in citytracker.keys(): 
        print(pcity + ': ' + str(citytracker[pcity]))
    else:
        print('0')

# looping through citytracker keys.
for city in citytracker.keys():
    count = citytracker[city]   # store the population in count
    print(city + ',' + str(count))    #print out key and values separated by a comma
import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here

import csv         # add csv module
w = csv.writer(open("popreport.csv", "w"))        # create file to write popreport.csv
w.writerow(['city','pop'])             #write column headers as city and population
for key, val in citytracker.items():           #loop through each key and value in citytracker
    w.writerow([key, val])              #write each value on new line

import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("geoAPI_key") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("weatherAPI_key") #replace "openweathermap" with the key name you created!

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
    #print(url)    #commented this out since it will make output easier to read.
    return safeGet(url)

data = json.load(getForecast())
print(data)
current_time = datetime.datetime.now() 

print(f"The current weather in Seattle is: {data['weather'][0]['description']}")
print("Retrieved at: %s" %current_time)

### You can add your own code here for Steps 2 and 3

print('\n')    # add space to separate from previous outputs

weatherforcity = {}   # create new dictionary


#loop through citytracker dictionary
for city in citytracker.keys():
    results = geocoder.geocode(city)     #define results this time taking in each city
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
    data = json.load(getForecast(city))     #getForecast data iterating through all cities, not just Seattle
    print (f"{city} is located at:")
    print (f"Lat: {lat}, Lon: {lng}")
    print(f"The current weather is: {data['weather'][0]['description']}")
    print("Retrieved at: %s" %current_time)
    weatherforcity[city]=json.load(getForecast(city))  #add getForecast data for all cities to new dictionary
    print('\n')



# create citytrackerweather.json file and use dump method to dump weatherforcity dictionary data to json file.
with open('citytrackerweather.json', 'w') as outfile:
    json.dump(weatherforcity,outfile)
    print('saved file')



    
