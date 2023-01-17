citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle'] = citytracker['Seattle'] + 17500

print(str(citytracker['Seattle']) + " residents in Seattle")
citytracker['Los Angeles'] = 45000

print(citytracker['Los Angeles'])
print("Denver: " + str(citytracker['Denver']))
for city in citytracker: 

    print(city, citytracker[city])
for city in citytracker: 

    print(city, ":", str(citytracker[city]))
nameTest = 'New York'

if nameTest in citytracker:

    print(nameTest,":",citytracker[nameTest])

else: 

    print("Sorry,", nameTest, " is not in the City Tracker.")

    

nameTest = 'Atlanta'

if nameTest in citytracker:

    print(nameTest,":",citytracker[nameTest])

else: 

    print("Sorry,", nameTest, " is not in the City Tracker.")

    
potentialcities = ['Cleveland','Denver','Phoenix','Nashville','Philadelphia','Milwaukee']

# added denver in slot[1] to make sure it was looping through

for city in potentialcities:

    if city in citytracker:

        print(city,":",citytracker[city])

    else: 

        print("Sorry,", city, " is not in the City Tracker.")
for city in citytracker: 

    print(city, ",", str(citytracker[city]))
import csv

with open('popreport.csv', 'w') as f:

    f.write("City Name,Population\n")

    for key in citytracker.keys():

        f.write("%s,%s\n"%(key,citytracker[key]))

    
import os



### Add your code here

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
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("DarkSkyAPI") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



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



# lat and lon below are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_1

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



data = json.load(getForecast(lat="47.656648",lng="-122.310233"))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])

for city in citytracker: 

    query = city

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (city, ":","Lat: %s, Lon: %s" % (lat, lng))

 



    data = json.load(getForecast(lat,lng))

   # print("Retrieved at: %s" %current_time)

   # print(data['currently']['summary'])

   # print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])
# I struggled on the synchronization of the output. I'm writing to the file, but it's not a super-clean one

# I went through a variety of ways to add headers, but this was the closest output adding headers kept breaking the file 



with open('cityreport.csv', 'w') as f:

    for item in data.items():

        for city in citytracker:

            f.write(city+","+str(item)+",\n")