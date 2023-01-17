citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago'])
newValue = 17500

citytracker['Seattle'] += newValue



print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000

print(citytracker['Los Angeles'])
print("Denver: %d" %citytracker['Denver'])
for city in citytracker.keys():

    print(city)
for city in citytracker.keys():

    print("%s: %d" %(city, citytracker[city]))
if 'New York' in citytracker.keys():

    print("New York: %d" %citytracker['New York'])

else:

    print("Sorry, New York is not in City Tracker.")

    

#Repeating the same process for Atlanta

if 'Atlanta' in citytracker.keys():

    print("Atlanta: %d" %citytracker['Atlanta'])

else:

    print("Sorry, that is not in City Tracker.")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities:

    print("%s: %d" % (city, citytracker.get(city, 0)))

for city in citytracker.keys():

    print('%s,%d'%(city, citytracker[city]))
import os

import csv

### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here

w = csv.writer(open("popreport.csv", "w"))

w.writerow('City, Population')

for city, pop in citytracker.items():

    w.writerow([city, pop])
import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("openWeatherMap") #replace "openweathermap" with the key name you created!



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

    return safeGet(url)



data = json.load(getForecast())

print(data)

current_time = datetime.datetime.now() 



print(f"The current weather in Seattle is: {data['weather'][0]['description']}")

print("Retrieved at: %s" %current_time)



### You can add your own code here for Steps 2 and 3
#Create an empty dict that would contain citites as keys and their forecast details as values

weatherTracker = {}

#For every city, add the forecase as value from our already created 'getForecast' function

for city in citytracker.keys():

    weatherTracker[city] = json.load(getForecast(city))



#Printing details of a city to check whether they exist in the dictionary    

print(weatherTracker['Chicago'])

with open('weatherDataJSON.txt', 'w') as outfile:

    json.dump(weatherTracker, outfile)