citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago']) # print call for citytracker dictionary using key Chicago to print corresponding value
print(citytracker['Seattle']) # get initial value of Seattle
citytracker['Seattle'] = citytracker.get("Seattle", 0) + 17500 # check the value of Seattle or return 0 if it doesn't exist, then add 17500 as requested, then reassign as the value for the Seattle key
print(citytracker['Seattle']) # get new value of Seattle
citytracker.setdefault('Los Angeles', 45000) # add a new key named Los Angeles with an initial value of 45000

print(citytracker['Los Angeles']) # check that there are no problems after adding
denverStr = "Denver: " + str(citytracker.get('Denver', 0)) # create a string for Denver data; typecast the value to a str for concatenation
print(denverStr) # print out the string
for city in citytracker.keys(): # traverse through the keys e.g., cities available in the citytracker
    print(f"{city} is a city in our dictionary") # print out the names/keys
for city in citytracker.keys(): # traverse through the keys
    print(f"{city}: {citytracker.get(city)}") # format and print the key and corresponding value
city1 = 'New York' # create a variable to hold our first city as a string
city2 = 'Atlanta' # do the same for second city

check1 = citytracker.get(city1, 0) # assign the value to another variable so we don't need to run the method multiple times
if check1 != 0: # if it returns 0, then it doesn't exist
    print(f"{city1}: {str(check1)}")
else:
    print(f"Sorry, {city1} is not in the City Tracker.")

# repeat for second city
check2 = citytracker.get(city2, 0)
if check2 != 0:
    print(f"{city2}: {str(check2)}")
else:
    print("Sorry, that is not in the City Tracker.")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities: # iterate through the list
    check = citytracker.get(city, None) # assign the value to a variable
    if check != None: # if it checks, then print out our formatted string with actual value
        print(f"{city}: check") 
    else: # otherwise, print formatted string with 0 as the value
        print(f"{city}: 0")
for city in citytracker.keys(): # get all keys
    print(f"{city},{citytracker.get(city)}") # print out CSV formatted keys + associated values
import os

### Add your code here
file = open("popreport.csv", "w") # open a file to write; since file does not exist, automatically writen

file.write("city,pop") # create a header row in the file

for city in citytracker.keys(): # get all keys from the dictionary
    file.write(f"\n{city},{citytracker.get(city)}") # write keys and associated value to file with a new line for each.

file.close() # close the file as part of good practice

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenCage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("OpenWeather") #replace "openweathermap" with the key name you created!

from opencage.geocoder import OpenCageGeocode

for query in citytracker.keys(): # traverse through the city names as pass them in as the query iteratively
    geocoder = OpenCageGeocode(secret_value_0) # make an API call
    results = geocoder.geocode(query) # assign the results of the call 
    lat = str(results[0]['geometry']['lat']) # parse out the latitude of the results
    lng = str(results[0]['geometry']['lng']) # parse out the longitude of the results
    print (f"{query} is located at:") # print our city
    print (f"Lat: {lat}, Lon: {lng}") # print the associated coordinates
import urllib.error, urllib.parse, urllib.request, json, datetime, os # import necessary packages; added os

# pre-written function by Brock
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

# pre-written function by Brock, modified to take arguments other than Seattle
def getForecast(city):
    key = secret_value_1
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)

cityForecasts = {} # create an empty dictionary to store forecast data

for city in citytracker.keys(): # use our existing dictionary to add 
    data = json.load(getForecast(city)) # make the API call to get forecast info and convert to a dictionary
    cityForecasts.setdefault(city, data) # assign key and value into our new dictionary
    print(data) 

for city in cityForecasts.keys(): # traverse our new forecasts dictionary
    print(f"The current weather in {city} is: {cityForecasts[city]['weather'][0]['description']}") # see what the current weather is in each city
    
### Pre-existing code written by Brock with slight modifications to write data to a file
file = open("citydata.json", "w") # open a json file to write; since file does not exist, automatically writen

for city in cityForecasts.keys(): # traverse our forecasts dictionary
    my_json = json.dumps({city:cityForecasts[city]}) # convert the key + value (which is a dictionary) to a json string for writing
    file.write(my_json) # write the json string to the file

file.close() # close the file as part of good practice

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))