citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#printing number of residents in Chicago

print(citytracker["Chicago"]) #printing value for key (Denver) within dictionary.
#increasing Seattle value by 17500
#Everytime I run this code it is increasing the value by 17500 and changes it in the dictionary


citytracker["Seattle"] = citytracker["Seattle"] + 17500 #Changing value of existing key to increase numnber or residents
print(citytracker["Seattle"]) #printing the new value


#creating a new key value pair
la = "Los Angeles" #Assigning the string Los Angeles to a variable
citytracker[la] = "45000" #creating a value for that key
print("The value assocated with Los Angeles is " + citytracker[la])
den = "Denver" #giving variable to city name
print("Denver: " + str(citytracker[den])) #printing string of population data
#printing a list of the keys

for cities in citytracker: #use a for loop to find keys in dictionary
    print(cities)
for key,val in citytracker.items(): # iterating through the dictionary looking for the key and the associated value
    print(key, ":", val) # separating the key and the value with an added colon

#I also tried it with a alternative print function
for key, value in citytracker.items():
    print(f"{key}: {value}") #using a differt print funciton to print key and associated value
if "New York" in citytracker:  #Looking for New York in the dictionary
    print("New York:" + str(citytracker["Atlanta"])) #if this statement is true it will print the key and the value
else: 
    print("Sorry, that is not in the City Tracker")  #if the statement is false it will print this statement
    

    
if "Atlanta" in citytracker:   #repeated code to test of Atlanta
    print("Atlanta:" + str(citytracker["Atlanta"]))
else: 
    print("Sorry, that is not in the City Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']


for city in potentialcities: #looping througt the potentialcities list
    if city in citytracker: #if the city is also in city tracker
        print(city + ": " + str(citytracker[city])) #print key and value
    else:
        print("zero") #print "zero" if the city doesn't exist
        
#I'm unsure if this is what you are looking for or if I still need to remove the spaces before and after the commas. I attempted to use the strip function to do so, but was unsucessful.

for key,val in citytracker.items(): # iterating through the dictionary looking for the key and the associated value
    print(key,",",val) # separating the key and the value with a comma 
for key, value in citytracker.items(): # for any key and value in citytracker   
    print(f"{key},{value}") #print the key and the value seperated by a comma.
import os

### Add your code here
 

with open("popreport.csv", 'w', newline='') as file: #opening the file you want to write to with a blank new line an assigning it to file
    writer = csv.writer(file)  #found this line of code online that I think assigns the variable to write to a row in the file
    writer.writerow(['city','pop']) #adding header row to file
  
    for key,val in citytracker.items(): #looking in citytracker for keys and values
        writer.writerow([key,val]) #writing each key and value to separate coloumn in the row.

file.close()
    

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("cage") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("weather") #replace "openweathermap" with the key name you created!

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(secret_value_0)
query = 'Salt Lake City'  # replace this city with cities from the names in your citytracker dictionary
results = geocoder.geocode(query)
lat = str(results[0]['geometry']['lat'])
lng = str(results[0]['geometry']['lng'])
print (f"{query} is located at:")
print (f"Lat: {lat}, Lon: {lng}")
#I lef this example code here for Seattle for reference and added the new code in cell below.
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


new_citytracker = {}    
for item in citytracker: #for each item in the citytracker dictionary 
    def getForecast(city=item): #defining the city as an item so that all cities data will be pulled from the database.
            key = secret_value_1
            url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
            print(url)
            return safeGet(url)
           

    data = json.load(getForecast())
    print(data)
    current_time = datetime.datetime.now() 

    
    print(f"The current weather in city is: {data['weather'][0]['description']}")
    print("Retrieved at: %s" %current_time)
    




new_citytracker = {} #creating new dictionary to store old population and new data in a nested dictionary
for item in citytracker: #for cities in the citytracker dictionary
    new_citytracker[item] = {"population": citytracker[item], "forecast": data}  #create new dictionary with value as city and includes population followed by forecast data dictionary 
print(new_citytracker) 
import json

with open('new_citytracker.json', 'w') as outfile: #writing new dictionary to a json file
    json.dump(new_citytracker, outfile) #using dump function to place data in file

    