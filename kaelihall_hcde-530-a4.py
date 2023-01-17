citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago'])#Search for the value of Chicago in the dictionary "citytracker"
citytracker['Seattle']=(citytracker['Seattle']+17500)#Change the value of key 'Seattle' by adding 17500 to the original value

print(citytracker['Seattle'])#Print new value
citytracker['Los Angeles']=45000#Adds a new key:value pair to the dictionary

print(citytracker['Los Angeles'])
print('Denver:%d' %citytracker['Denver'])

#Signal to python with the % that the string will call an integer

#Then define the integer with a second % by calling the value in the dictionary
cities=citytracker.keys()#Define variable for a list of just the keys from the citytracker dictionary

for citynames in cities:#Loop that variable of keys

    print(citynames)
for citynames in cities:

    print(citynames,": %d" %citytracker[citynames])

#Uses the % to call the value of the key input by the loop "citynames"
if 'New York' in citytracker: #Searches for value New York in city tracker dictionary

    print('New York : %d' %citytracker['New York'])#If true, will print New York and its value

else:

    print("Sorry that is not in the City Tracker")#If false, will print message informing that it's not in the dictionary



#Same formula below with input for "Atlanta"

if 'Atlanta' in citytracker:

    print('Atlanta : %d' %citytracker['Atlanta'])

else:

    print("Sorry that is not in the City Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for search in potentialcities:#Loops the new list

    if search in citytracker:#Checks if strings in the new for loop exist in dictionary citytracker

        print(search,' : %d' %citytracker[search])#If true, prints city:population

    else:

        print(search, ": 0")#Otherwise, prints zero
for citynames in cities:#Same loop as question 6

    print(citynames+",%d" %citytracker[citynames])#Slight change in code, using a + instead of , between the loop and %integer, so that there will not be a space in the print

    #Also replaced the ":" with a ","
import os



### Add your code here

import csv

with open('popreport.csv','w', newline='') as outfile:

    thewriter=csv.DictWriter(outfile, fieldnames=['City','Pop'])#Assigns a variable to the writer that I am using to creat the CSV

    thewriter.writeheader()

    for citynames in cities:

        thewriter.writerow({'City':citynames, 'Pop':citytracker[citynames]})



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage") # make sure this matches the Label of your key

key1 = secret_value_0

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))

print(cities)

def latlong(x):#Create a procedure to find the lattitudes and longitudes for each city

    from opencage.geocoder import OpenCageGeocode

    geocoder = OpenCageGeocode(key1)

    query = (x)  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print ("lat=%s,lng=%s" %(lat, lng))

latlong('Atlanta')

latlong('Boston')

latlong('Milpitas')

latlong('Winston-Salem')

latlong('Hanoi')

for locations in cities:

    print (locations)

    latlong(locations)

# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSky") # make sure this matches the Label of your key



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



#getForecast(lat="33.7490987",lng="-84.3901849")

#getForecast(lat="42.3602534",lng="-71.0582912")

#getForecast(lat="41.8755616",lng="-87.6244212")

#getForecast(lat='39.7392364',lng='-104.9848623')

# lat and lon below are for UW

def getForecast(lat=x,lng=y): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



getForecast(x="33.7490987", y="-84.3901849")



data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])

import json



def getForecast(lat="33.7490987",lng="-84.3901849"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)







data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])





def getForecast(lat="42.3602534",lng="-71.0582912"):

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)







data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



getForecast(lat="33.7490987",lng="-84.3901849")

data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



getForecast(lat="42.3602534",lng="-71.0582912")

data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



getForecast(lat="41.8755616",lng="-87.6244212")

data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



getForecast(lat='39.7392364',lng='-104.9848623')

data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



with open("cityweather.txt", 'w') as outfile:

    json.dump(data,outfile)