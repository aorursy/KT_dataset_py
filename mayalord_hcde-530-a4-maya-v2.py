citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker["Chicago"])



#printing the number of residents in Chicage

print(citytracker["Seattle"])



citytracker["Seattle"]  = citytracker["Seattle"] +  17500



print(citytracker["Seattle"])



#changes the number of Seattle residents and printing number of residents
citytracker["Los Angeles"] = 4500

print(citytracker["Los Angeles"])



#adding LA to dict and printing number of residents
d  = "Denver: " + str(citytracker["Denver"])



print(d)



#prints string with number of residents
for city in  citytracker.keys():

    print(city)

    

#using loop, going through all cities printing each
for city in citytracker.keys():

    print(city + ":", citytracker[city])

    

##using loop, going through all cities printing each with the number of residents
if 'New York' in citytracker:  

    print("New York:  " + str(citytracker['New York']))

else: 

    print("Sorry, that is not in the City Tracker")

    

if 'Atlanta' in citytracker:  

    print("Atlanta:  " + str(citytracker['Atlanta']))

else: 

    print("Sorry, that is not in the City Tracker")

    

    

## if statement, if the city is in the dict, it gives the name  and number of  residents, else it states it isnt' in the dict
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']





for pcity in potentialcities: 

    if pcity in citytracker: 

        print(pcity + ':' + str(citytracker[pcity]))

    else:

        print(0)

        

##Loops through potentialcities checking if they are in the  citytracker dict.  

## if yes they print the name and number of residents  

## if  not print 0
for city in citytracker.keys():

    print(city + "," + str(citytracker[city]))

    

##loops from citytracker dict, printing the name of  city and number of residents
import os



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here



import csv

        

with open("popreport.csv","w") as file: 

    writer = csv.writer(file)

    writer.writerow(["city", "pop"])

    

    for city in citytracker.keys():

        writer.writerow([city, str(citytracker[city])])

        

##imports csv

##writes file with name popreport.csv with two columns city and pop

## and files in the columns

        
import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("openWeather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = city  # replace this city with cities from the names in your citytracker dictionary

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









cityDict  = {}



##empty dict to store new combined information



for city in citytracker.keys(): #loops through citytracker dict for each city 

    

    results = geocoder.geocode(city)

    data = json.load(getForecast(city))      ##gets the weather data from opencage

    lat = str(results[0]['geometry']['lat']) ##gets longitude info from openweather

    lng = str(results[0]['geometry']['lng'])  ##gets latitude info from openweather

    

    print (f"{city} is located at:")         ##prints city name for each

    print (f"Lat: {lat}, Lon: {lng}")         ##print the latitude and longitude

    print(f"Currently the weather is: {data['weather'][0]['description']}") ##prints the weather for each city

   

    cityDict[city] = {"Latitude": lat, "Longitude": lng, "Weather": data,}  ##add information for  each city into dict



    

with open ("cityDict.json",  "w") as outfile:

      json.dump(cityDict, outfile)

        

##defines file to write and saves as json