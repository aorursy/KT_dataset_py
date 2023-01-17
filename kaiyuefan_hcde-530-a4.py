citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

citytracker['Chicago']
citytracker['Seattle']=724725+17500

citytracker['Seattle']
citytracker['Los Angeles']=45000

citytracker['Los Angeles']
x=citytracker['Denver']

print('Denver: '+str(x))
for x in citytracker.keys():

    print(x)
for x in citytracker.keys():

    print(x+': '+str(citytracker[x]))
if 'New York'in citytracker:

    print('New York: '+str(citytracker['New York']))

else:

    print('Sorry, that is not in tehe City Tracker.')



if 'Atlanta'in citytracker:

    print('Atlanta: '+str(citytracker['Atlanta']))

else:

    print('Sorry, that is not in tehe City Tracker.')
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for x in potentialcities:

    if x in citytracker:

        print(x+': '+str(citytracker[x]))

    else:

        print(0)
for x in citytracker.keys():

    print(x+','+str(citytracker[x]))
import os



### Add your code here

import csv



with open('popreport.csv', mode='w') as file:

    file_writer = csv.writer(file, delimiter=',')

    file_writer.writerow(['city', 'pop'])

    for x in citytracker.keys():

        file_writer.writerow([x, str(citytracker[x])])





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencagekey") # make sure this matches the Label of your key

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

secret_value_0 = user_secrets.get_secret("darkskykey") # make sure this matches the Label of your key



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



# lat and lon below are for UW

def getForecast(lat="47.6038321",lng="-122.3300624"): # default values are for UW

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





citytemp={}



citytemp['Atlanta']=63.49

citytemp['Boston']=40.99

citytemp['Chicago']=48.17

citytemp['Denver']=81.87

citytemp['Seattle']=41.19



import json

with open('citytemp.json', 'w') as outfile:

    json.dump(citytemp, outfile)
