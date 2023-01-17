citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker ["Chicago"])
citytracker ['Seattle'] = citytracker['Seattle'] + 17500

#replacing the current value for the key seattle with the value + 17500

print (citytracker ['Seattle'])

citytracker ["Los Angeles"] = 45000 #add a new item

citytracker #printing the value
city = "Denver" # defining city as Denver 
city_pop = citytracker[city] #creating new variable to hold value
string = city + ":" + str (city_pop) # Defining string as key plus the value
print (string) #print the value as a string
for key in citytracker: #for loop looking at each item in the dictionary
    print (key)
for key in citytracker: #assign city as the key and iterate through each one
    print (key +":"+ str(citytracker[key])) #print the key plus the value
if "New York" in citytracker: #if statement to test whether New York is in citytracker
    print ("New York:" + str (citytracker["New York"])) # If yes then print key and value
else:
    print ("Sorry, that is not in the City Tracker") # if not then print this
    
# same for below
    
if "Atlanta" in citytracker: 
    print ("Atlanta:" + str(citytracker["Atlanta"]))
else:
    print ("Sorry, that is not in the City Tracker")

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities: #looping through the list
    if city in citytracker:
        print (city + ":" + citytracker [city]) # #prints the city name and the value if the key exists
    else:  # otherwise shows city name and default value
        print (city +":0")
        
        

for key in citytracker: #assign city as the key and iterate through each one
    print ( key + "," + str (citytracker[key])) # #print the key plus the value with comma and no spaces
import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import csv
        
citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
w = csv.writer(open("popreport.csv", "w"))
w.writerow(["city", "pop"]) #write the first row to be a header
for key, value in citytracker.items(): #iterate through city tracker keys
    w.writerow([key,value]) #use writer and writerow method to add each key and it's value
    
    
    # I see only the header in my csv file but no data. Not sure why 
    


import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Open Cage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("Open Weather") #replace "openweathermap" with the key name you created!

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

### You can add your own code here for Steps 2 and 3

forecast = {} #create new dictionary called forecast 

for city in citytracker: #iterate through city tracker keys
    data = json.load(getForecast(city)) #created new data variable to store data pulled from each city using getForecast module
    forecast [city] = data # store data variable inside newly created forecast dictionary where they key is the city and the value is the pulled data 
    
import json # copy and pasted code from earlier help article

json = json.dumps(forecast) #dumping forecast dictionary to json
f = open("forecast.json","w") #opening json file to give write ability
f.write(json) #f is document, I am going to write whatever in JSON to document, like copy and paste 
f.close()


