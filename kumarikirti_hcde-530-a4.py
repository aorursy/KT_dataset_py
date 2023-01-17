citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago'])
citytracker['Seattle']=17500

print(citytracker)

print(citytracker['Seattle'])
citytracker['Los Angeles']=45000

print(citytracker)

print(citytracker['Los Angeles'])
x=str(citytracker['Denver'])

print("Denver: "+x)
for key in citytracker:

    print(key)
for key,value in citytracker.items():

    print(str(key)+":"+str(value))

    
if 'New York' in citytracker.keys():

        print("New York"+str(citytracker['New York']))

else:

     print("Sorry, that is not in the Coty Tracker")

        

if 'Atlanta' in citytracker.keys():

        print("Atlanta"+":"+str(citytracker['Atlanta']))

else:

     print("Sorry, that is not in the Coty Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



def citycheck(key):

  if key in citytracker.keys():

        print(key+":"+str(citytracker[key]))

  else:

     print("Sorry, that is not in the Coty Tracker")



for x in potentialcities:

    citycheck(str(x))
for x in citytracker.items():

    print(x)
import os



### Add your code here







### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencageKey") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

#print(results)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
for key in citytracker.keys():

    query2=str(key)

    results = geocoder.geocode(query2)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key



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

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

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
