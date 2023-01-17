citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(f"Chicago: { citytracker['Chicago'] }")
for word in citytracker:

    if word == 'Seattle':

        citytracker[word] = citytracker.get(word)+17500

print(f"Seattle: { citytracker['Seattle'] }")
citytracker.update( {'Los Angeles' : 45000} )

print(f"Los Angeles: { citytracker['Los Angeles'] }")


x= sum(map(int, citytracker.values()))

print(f"Denver: "+ str(x))
for word in citytracker.keys():

        print(word)
for word in citytracker:

    print (word, citytracker[word])
for word in citytracker.keys():

    if word == 'New York':

        print (word, citytracker[word])

    else:

        print ("Sorry, that is not in the city tracker")

   

    if word == 'Atlanta':

        print (word, citytracker[word])

    else:

        print("Sorry, that is not in the city tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for word in potentialcities:

    if word in citytracker.keys():

        print(word, citytracker[word])

    else:

        print('0')
for word in citytracker:

    print (word + "," + str(citytracker[word]))
import csv

import os

    

### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here

w = csv.writer(open("popreport.csv", "w"))

fieldnames = ['city', 'pop']

w.writerow(fieldnames)

for key, val in citytracker.items():

    w.writerow([key, val])

#I added the code above.

import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("openweathermap") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = word # replace this city with cities from the names in your citytracker dictionary

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



def getForecast(city='Seattle'):

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



for city in citytracker.keys():

    population = citytracker[city]

    citytracker.update({city:{"population": population, "forecast": {"lat":45, "log":99}}})

print(citytracker)



with open('citydata.json', 'w') as outfile:

    json.dump(citytracker, outfile)

print(citytracker)

with open('citydata.json', 'w') as outfile:

    json.dump(citytracker, outfile)