citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
# We print the population of Chicago
print(citytracker['Chicago'])
# Add 17,500 residents to Seattle
citytracker['Seattle']+=17500
# We print the population of Seattle
print(citytracker['Seattle'])
# Add Los Angeles with 45,000.
citytracker['Los Angeles']=45000
# We print the population of Los Angeles.
print(citytracker['Los Angeles'])
# Our Denver string
denver_res = 'Denver: ' + str(citytracker['Denver'])
# We print denver_res
print(denver_res)
# Iterate through the dictionary keys(no need to use "keys()"" according to https://www.python.org/dev/peps/pep-0234/) and print them
for city in citytracker:
    print(city)
# Iterate through the dictionary items and print them
for city,pop in citytracker.items():
    print(city + ' : ' + str(pop))
# Function that receives a city and prints its population if it exists in out dictionary, otherwise, it lets the caller know it doesn't
def get_population(city):
    if city in citytracker:
        print(city + ' : ' + str(citytracker[city]))
    else:
        print('Sorry, that is not in the City Tracker,')

# Check answers
get_population('New York')
get_population('Atlanta')
        

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

# Iretate through the potential cities
for city in potentialcities:
    # Print it's population checking in citytracker
    pop = citytracker.get(city,0)
    print(city + ' : ' + str(pop))
# Iterate through the dictionary items and print them separeted by "comma"
for city,pop in citytracker.items():
    print(city + ',' + str(pop))
import os
import csv

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here
# Open our file using "with", https://docs.python.org/3/reference/compound_stmts.html#the-with-statement so that we can forget about closing it
with open('popreport.csv','w') as f: 
    # Define our writer
    w = csv.writer(f)
    # Write the first row(headers)
    w.writerow(['City', 'Pop'])
    # Write all the dictionary items as rows
    w.writerows(citytracker.items())

# Open the file again to check its content
with open('popreport.csv') as f:
    # Define our reader
    r = csv.reader(f)
    print('\nCity Tracker(CSV):\n')
    # Iterate through each row and print it
    for row in r:
        print(row)

###############################################
# Step 0
###############################################

print("##############################################")
print("Step 0 - Install OpenCage")
print("##############################################")
!pip install opencage # Install OpenCage package

###############################################
# Step 1
###############################################

print("\n##############################################")
print("Step 1 - Location Information")
print("##############################################")

import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("opencage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("openweather") #replace "openweathermap" with the key name you created!

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
    except urllib.error.URLError as e:
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

###############################################
# Step 1.5 helper functions
###############################################

# Function that gets the geocode information for the given city. It returns latitude and longitude as a dictionary.
def getGeocode(city):
    gcinfo = geocoder.geocode(city)
    lat = str(gcinfo[0]['geometry']['lat'])
    lng = str(gcinfo[0]['geometry']['lng'])
    return {'lat': lat, 'lng': lng}

forecast = {}
# Function that returns the result of getForcast in an json
def getForcastResult(city):
    return json.load(getForecast(city))

# Function that gets the weather information for the given city. It returns type(weather) and description as a dictionary.
def getWeather(city):
    # If we don't have forecast for this city, get it from the API and save it to our forecast dictionary
    if city not in forecast:
        forecast[city]=getForcastResult(city)
    # Get the weather info only
    weather_forecast = forecast[city]['weather']
    return {'weather' : weather_forecast[0]['main'], 'weather_desc' : weather_forecast[0]['description']}

# Function that gets the temperature information for the given city and forcast info. It returns temperature in Celcius(c)
def getTemperature(city):
    # If we don't have forcast for this city, get it from the API and save it to our forcast dictionary
    if city not in forecast:
        forecast[city]=getForcastResult(city)
    # Get the temperature info only
    temp_forecast = forecast[city]['main']
    # Float for decimals and round to only get 2 decimals
    return round(float(temp_forecast['temp'])-273.15, 2)

###############################################
# Step 2
###############################################

print("\n##############################################")
print("Step 2 and Step 3 - Call the methods in section 1.5 and print our results")
print("##############################################")

# Iterate through the dictionary items and print them
for city in citytracker:
    # Get the Geocode data
    geocode = getGeocode(city)
    # Get the weather data
    weather = getWeather(city)
    # Get the temperature
    temperature = getTemperature(city)
    
    # Print them all to confirm(I'm commenting these prints to keep my output clean)
    # print(geocode)
    # print(weather)
    # print(temperature)
    
    # Temporarily save the population because we are changing the format of our dictionary
    population = citytracker[city]
    # Save all the info to out dictionary in our new format {city: {pop,temp,lat,long,weather,weather_desc}}
    citytracker[city]={'pop':population, 'temp': temperature}
    citytracker[city].update(geocode)
    citytracker[city].update(weather)

# Print out result JSON(in pretty mode)
json_citytracker = json.dumps(citytracker, indent=2)
print(json_citytracker)

###############################################
# Step 3
###############################################

# Create a new JSON file and write out citytracker json to it
f = open("citytracker.json","w")
f.write(json_citytracker)
f.close()
