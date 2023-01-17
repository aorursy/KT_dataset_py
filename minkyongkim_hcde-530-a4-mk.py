citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle'] = citytracker['Seattle'] + 17500 # add 17500 to value of Seattle

print (citytracker) 

    
citytracker['Los Angeles'] = 45000 # set value of Los Angeles to 45000

print(citytracker)
print("Denver: " + str(citytracker['Denver'])) 
for city in citytracker.keys(): # for each city in the dictionary

    print(city) # print the city
for key in citytracker.keys():

    print (key,":",citytracker[key])
if 'New York' in citytracker:

    print("New York",":", citytracker['New York']) # if New York is in citytracker dictionary

elif 'Atlanta' in citytracker: # if false then try looking for Atlanta 

    print("Atlanta",":", citytracker['Atlanta']) # if found, print the value for Atlanta

else:

    print("Sorry that is not in the City Tracker")  # if Atlanta is not found, print "Sorry that is not in the City Tracker"
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
for city in potentialcities: # for each city in potentitalcities dictionary

    if city in citytracker: # if that city is also in citytracker dictionary

        print(city, ":", citytracker[city]) # print the city with its population value

    else:

        print(0) # if not in the citytracker dictionary, print 0
for key in citytracker.keys(): # for each key in the citytracker keys

    print(str(key) + "," + str(citytracker[key])) # print key value pair separated by commas
import os

### Add your code here

import csv



with open ('popreport.csv', 'w', newline ='') as file:

    fields = ['city', 'pop']

    w = csv.writer(file) 

    w.writerow(fields) # write header row

    w.writerows(citytracker.items()) # fill rows with key value pairs



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage") # make sure this matches the Label of your key

key1 = secret_value_0



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darksky") # make sure this matches the Label of your key



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

    

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



temp = {} # create empty list for temperature values

citylist = list(citytracker.keys())  # create list out of just the cities

for city in citylist: # for each city in the list of cities

    results = geocoder.geocode(city) # store coordinates of city in results

    lat = str(results[0]['geometry']['lat']) # get the latitude value and store it in 'lat'

    lng = str(results[0]['geometry']['lng']) # get the longitude value and store it in 'lng'

    print (city, "Lat: %s, Lon: %s" % (lat, lng))

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    data = json.load(safeGet(url)) # get the data from above url using latitude and longitude coordinates

    temp[city] = data['currently']['temperature'] # inside the temp dictionary, store temp values for each city 



print(temp) # print new dictionary
#data = json.load(getForecast(lat,lng)) # commented out to just create dictionary of temp values

#current_time = datetime.datetime.now() 

#print("Retrieved at: %s" %current_time)

#print(data['currently']['summary'])

#print("Temperature: " + str(data['currently']['temperature']))

#print(data['minutely']['summary'])



import json

with open('temperature_values.json','w') as outfile:

    json.dump(temp, outfile)