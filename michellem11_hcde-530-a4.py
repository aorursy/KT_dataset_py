citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker['Chicago'])

#using .get() to increase the value for "Seattle" by 17500

citytracker['Seattle']=citytracker.get('Seattle',0)+17500

print(citytracker['Seattle'])
#add new key

citytracker['Los Angeles']=45000



#print new key value

print(citytracker['Los Angeles'])
#assign x to be the value of the key "Denver"

x=citytracker['Denver']

print("Denver:%s"%x)
for i in citytracker.keys():

    print(str(i))


for k in citytracker.keys():

     print(str(k)+': '+ str(citytracker[k]))
#testing if "New York" is in dictionary

if 'New York' in citytracker.keys():

    print ("New York: "+ str(citytracker["New York"]) )

else:

    print ("Sorry, that is not in the City Tracker.")



#testing if "Atlanta" is in dictionary

if 'Atlanta' in citytracker.keys():

    print ("Atlanta: "+ str(citytracker["Atlanta"]) )

else:

    print ("Sorry, that is not in the City Tracker.")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for i in potentialcities:

    if i in citytracker:

        print(str(i)+': '+ str(citytracker[i]))

    else: 

        print ("0")

for k in citytracker.keys():

     print(str(k)+','+ str(citytracker[k]))
import os

import csv



### Add your code here

  

with open("popreport.csv","w",newline='') as blankfile:

    w = csv.DictWriter(blankfile, fieldnames = ["city", "pop"])

    w.writeheader()

    for line in citytracker:

        w.writerow({"city":line,"pop":citytracker[line]})  



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencageAPI") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'New York'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkskyAPI") # make sure this matches the Label of your key



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

#get the lat & lng of the cities in citytracker

cityinfo={}

for i in citytracker.keys():

    results = geocoder.geocode(i)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])



    #use those coordinates to get forcast for those cities    

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now()

    #create a new dictionary to contain forcast info for each city

    cityinfo[i]={"Time":"Retrieved at: %s" %current_time,

                "Temperature":data['currently']['temperature'],

                "MinutelyForcast":data['minutely']['summary'],

                "CurrentlyForcast":data['currently']['summary'] }



    print("City: %s" %str(i))

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])    



    

print(cityinfo)
import json



with open('cityinfo.json', 'w') as fp:

    json.dump(cityinfo, fp)
