citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker['Chicago'])



#printed 'chicago' key to find the value
citytracker['Seattle'] = 17500

print(citytracker)



# Added the key 'Seattle' and printed the citytracker dictionary.
citytracker['Los Angeles'] = "45000"

print(citytracker['Los Angeles'])



# Assigned a value to Los Angeles and printed the key
for key in citytracker:

    print(key)

    

# for loop to print each city of the key values.
for key, value in citytracker.items:

    print(key, 'corresponds to', value)

    

# I found 'corresponds to' on google to print these code
if "New York" in citytracker:

    print("New York : <x>")

else:

    print("Sorry, that is not in the Coty Tracker.")

    

if "Atlanta" in citytracker:

    print("Atlanta : 486290")

else:

    print("Sorry, that is not in the Coty Tracker.")

    

# If, else statment to test if the key is in the dictionary
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
if potentialcities in citytracker:

    print("City : #")

else:

    print('0')

    

#I wanted to print potentialcities with if statement, but it did not work.
# I was not able to get step 6 so I am not sure how I can solve this
import os



### Add your code here







### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# Having a hard time figuring this out.
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey") # make sure this matches the Label of your key

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
