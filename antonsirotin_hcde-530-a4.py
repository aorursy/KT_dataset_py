citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#prints the value of dictionary key 'Seattle'
print(citytracker['Chicago'])
#adds 17,500 to the value of dict key 'Seattle'
citytracker['Seattle'] + 17500
#adds the new key 'Los Angeles', along with its value, to the dictionary 'citytracker' and prints the value associated with that key
citytracker['Los Angeles'] = 45000
print(citytracker['Los Angeles'])
#prints a string along with a value pulled from dict 'citytracker' after converting the value to a string
print('Denver: ' + str(citytracker['Denver']))
#iterates through dict 'citytracker' and prints each key name
for city in citytracker:
    print(city)
#iterates through dict
for city in citytracker:
    #prints a string that includes the key name and the value of each value iteration converted into a string
    print(city + ': ' + str(citytracker[city]))
#checks to see if citytracker contains New York key
#if it does, it prints the value of that key
#if it doesn't it prints a 'Sorry' message
if 'New York' in citytracker:
    print('New York:', citytracker['New York'])
else:
    print('Sorry, that is not in the City Tracker')
    
#checks to see if citytracker contains Atlanta key
#if it does, it prints the value of that key
#if it doesn't it prints a 'Sorry' message
if 'Atlanta' in citytracker:
    print('Atlanta:', citytracker['Atlanta'])
else:
    print('Sorry, that is not in the City Tracker')
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities:
    if city in citytracker:
        print(city + ': ' + str(citytracker[city]))
    else:
        print('0')
#iterates through dict
for city in citytracker:
    #prints a string that includes the key name and the value of each value iteration converted into a string, separated by a comma to represent collumn
    print(city + ',' + str(citytracker[city]))
import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here
#imports csv module
import csv
#write a file named popreport through writer method within csv module and name it as var f
f = csv.writer(open("popreport.csv", "w"))
#loop through key value tuples within dict citytracker using two iterables
#and write the key value as rows ofcomma separated values
for key, val in citytracker.items():
    f.writerow((key, val))

import urllib.error, urllib.parse, urllib.request, json, datetime, opencage

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("openCage")
secret_value_1 = user_secrets.get_secret("openWeather")

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(secret_value_0)
for gorod in citytracker:
    query = gorod  # replace this city with cities from the names in your citytracker dictionary
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

def getForecast(city="Paris"):
    #for cities in citytracker:
    key = secret_value_1
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)

#creates fdict dictionary to hold combination of forecast and citytracker data
fdict = {}

#iterates through citytracker dictionary
for cities in citytracker:
    #loads the output of getForecast into 'data'
    data = json.load(getForecast(cities))
    current_time = datetime.datetime.now() 
    print(f"The current weather in {cities} is: {data['weather'][0]['description']}")
    print("Retrieved at: %s" %current_time)
    
    #combines weather forecast from 'data' with the citytracker data
    fdict[cities] = [citytracker[cities], data['weather'][0]['description']]

#creates a csv file and writes into it the iterations of fdict
d = csv.writer(open('dictFile.csv', 'w'))
for city, l in fdict.items():
    d.writerow([city, l[0], l[1]])

#converts fdict into json object and writes o bject into json file
json = json.dumps(fdict)
f = open("fdict.json","w")
f.write(json)
f.close()


### You can add your own code here for Steps 2 and 3