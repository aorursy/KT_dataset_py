citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker['Chicago']) #prints the value assigned to the 'Chicago' key in the citytracker dictionary
newRez = citytracker['Seattle'] + 17500 #define a variable called newRez to add the new residents to the current value assigned to the key 'Seattle' in the dictionary citytracker
print(newRez)
citytracker['Los Angeles'] = 45000 #define the key and value for Los Angeles and its number of residents
print(citytracker['Los Angeles']) #print the value assigned to the 'Los Angeles' key within the citytracker dictionary
x = citytracker['Denver']
print('Denver: ' + str(x))
for key in citytracker: #for each key in citytracker
    print(key) #print the key
for key in citytracker.keys(): # for every key in the dictionary
    print (key + ": " + str(citytracker[key])) # print the key, then a colon and a space, and the value assigned to each key within the dictionary. Turn the value into a str so it can print.
x = citytracker[key] # define a variable x to hold the keys within the citytracker
newYork = str('New York') # create another variable named newYork to hold the value for the 'New York' key. Turn it into a string so it can print.
atLanta = str('Atlanta') # do the same for a variable named atLanta to hold the value for the 'Atlanta' key.
for key in citytracker.keys(): # for every key in the dictionary...
    if newYork == True: # if the key is 'New York'
        print('New York:' + ' ' + str(x)) # print the city and the value assigned to it
    else: # otherwise...
        print('Sorry, that is not in the City Tracker') # print an error message
    if atLanta == True: # repeat this process for atLanta/'Atlanta'
        print('Atlanta:' + ' ' + str(x)) # ''
    else: # ''
        print('Sorry, that is not in the City Tracker') #''

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
for potentialcities in citytracker:
    citytracker.get(potentialcities, 0)
    if potentialcities == citytracker.get(potentialcities):
        print(potentialcities, 0)
    else:
        print('zero')
        
for key, value in citytracker.items():
    print(key,':', value)
import os
import csv
    
### Add your code here
for key, value in citytracker.items():
  print(key, value)

# ok idk how to do this w/o pandas tbh...

print(df)
### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("openCageKey") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("openweathermap") #replace "openweathermap" with the key name you created!

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