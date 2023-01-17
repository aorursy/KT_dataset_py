citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Chicago']
#add 17500 to the original value of 'Seattle'

citytracker['Seattle'] += 17500

citytracker['Seattle']
#add new key&value

citytracker['Los Angeles'] = 45000

citytracker['Los Angeles']
denver = "Denver: "+ str(citytracker['Denver'])

print(denver)
for keys in citytracker.keys():

    print(keys)
for keys in citytracker.keys():

    print(keys + ":" + str(citytracker[keys]))
# check if NY is in the dict

if 'New York' in citytracker.keys():

    print ("New York: %s" %citytracker['New York'])

else:

    print ('Sorry, this is not in the City Tracker.')



# check if Atlanta is in the dict

if 'Atlanta' in citytracker.keys():

     print ("Atlanta: %s" %citytracker['Atlanta'])

else:

    print ('Sorry, this is not in the City Tracker.')        
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for cities in potentialcities:

    print (cities + ": %s" %citytracker.get('cities',"0"))
for keys in citytracker.keys():

    print(keys + "," + str(citytracker[keys]))
import os



### Add your code here



import csv



# pre-set a header for the csv file

header = ['city', 'pop']



# create a csv file

f = open('popreport.csv','w', newline="")

writer = csv.writer(f)



# add header row here

# got this code from google

# do not understand what 'i for i' means

writer.writerow(i for i in header)



# import dictionary keys and values

for key, value in citytracker.items():

    writer.writerow([key, value])



f.close()



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("OpenCage") # make sure this matches the Label of your key

key1 = secret_value_1



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
