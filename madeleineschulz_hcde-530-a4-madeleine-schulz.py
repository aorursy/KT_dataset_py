citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker["Chicago"])

citytracker["Seattle"]= 724725 + 17500

print(citytracker["Seattle"])
citytracker["Los Angeles"]= 45000

print(citytracker["Los Angeles"])
x = citytracker["Denver"]



print("Denver: %s" %x)
for line in citytracker:

    print(line)




for line in citytracker:

    y = citytracker[line]

    print(line +  ": %s" %y)
try:

    print("New York" + ": " + str(citytracker["New York"]))

except KeyError as detail:

    print("Sorry, that is not in the city tracker")



try:

    print("Atlanta" + ": " + str(citytracker["Atlanta"]))

except KeyError as detail:

    print("Sorry, that is not in the city tracker")    
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for item in potentialcities:

    try:

        print(item + ": " + str(citytracker[item]))

    except KeyError as detail:

        print(0)

        

        

#not sure where default values would come into play here? Unless I did this wrong...
for line in citytracker:

    y = citytracker[line]

    print(line +  ",%s" %y)
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

secret_value_0 = user_secrets.get_secret("openCageKey") # make sure this matches the Label of your key

key1 = secret_value_0

    

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



cityweather = {}



for line in citytracker:

    from opencage.geocoder import OpenCageGeocode

    geocoder = OpenCageGeocode(key1)

    query = line  # replace this city with cities from the names in your citytracker dictionary

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

    def getForecast(lat,lng): # default values are for UW

        # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    cityweather[line] = {"Currently": data['currently']['summary'],

                         "Temperature": data['currently']['temperature']}

                         #"Forecast": data['minutely'] I COULDN'T GET THIS INTO MY DICTIONARY

with open('citydata.txt', 'w') as outfile:

    json.dump(cityweather, outfile)



    

    





for line in citytracker:

    from opencage.geocoder import OpenCageGeocode

    geocoder = OpenCageGeocode(key1)

    query = line  # replace this city with cities from the names in your citytracker dictionary

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

    def getForecast(lat,lng): # default values are for UW

        # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    print(line + ":")

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary']+ "\n")

    
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
