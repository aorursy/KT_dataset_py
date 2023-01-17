citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle'] += 17500
print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000
print(citytracker['Los Angeles'])
print("Denver: " + str(citytracker["Denver"]))

for city in citytracker:
    print(city)
for city in citytracker:
    print(city + ": " + str(citytracker[city]))
def cityTester(city, citytracker):
    if city in citytracker:
        print(city + ": " + str(citytracker[city]))
        return;
    print("Sorry, that is not in the City Tracker.")
    return;

cityTester("New York", citytracker)
cityTester("Atlanta", citytracker)

#use of boolean to print true/false value
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
for city in potentialcities:
        print(city + ": " + str(citytracker.get(city, 0)))
        

#default value used if city does not exist within dataset, printing zero if cityname and value does not exist

for city in citytracker:
    print(city + "," + str(citytracker[city]))
import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here

f = open("popreport.csv", "a")

f.write("city,pop\n")
for city in citytracker:
    f.write(city + "," + str(citytracker[city]) + "\n")
f.close()




import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("label1") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("label2") #replace "openweathermap" with the key name you created!

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

for city in citytracker:
    from opencage.geocoder import OpenCageGeocode
    geocoder = OpenCageGeocode(secret_value_1)
    query = city # replace this city with cities from the names in your citytracker dictionary
    results = geocoder.geocode(query)
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
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
    def getForecast(lat,lng):
    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]
        key2 = secret_value_1
        url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
        return safeGet(url)

    data = json.load(getForecast(lat,lng))
    current_time = datetime.datetime.now() 
    
    # print the results of the API calls to verify things are working as expected
    print(city)
    print ("Lat: %s, Lon: %s" % (lat, lng))
    print("Retrieved at: %s" %current_time)
    print(data['currently']['summary'])
    print("Temperature: " + str(data['currently']['temperature']))
    print(data['minutely']['summary'])
    
    # update cityweather dictionary with key as city and then the data as the value
    cityweather = {
        city : {'latitude':lat, 
                'longitude':lng, 
                'retrieved at time':str(current_time), 
                'weather':data['currently']['summary'],
                'temperature':str(data['currently']['temperature']),
                'summary':data['minutely']['summary']}
    }
    
    # update the original dictionary with the cityweather dictionary
    citytracker.update(cityweather)

    # print the dictionaries to verify
    print(cityweather)
    print(" ")
    print(citytracker)
    print("")
    
    #not sure why API does not authorize for step2
    
    

#step 3

with open('jsonfile.json', 'w') as json_file:
        json.dump(citytracker, json_file)