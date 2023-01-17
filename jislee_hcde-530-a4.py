citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print ("Number of residents in Chicago: " + str(citytracker["Chicago"]))
citytracker["Seattle"]+=17500

print (citytracker["Seattle"])
citytracker["Los Angeles"]=45000

print(citytracker["Los Angeles"])
x=citytracker["Denver"]

print("Denver: "+ str(x))
for x in citytracker:

    print(x)
for x,y in citytracker.items():

    print(str(x)+ ":"+ str(y))
if "New York" in citytracker:

    print ("New York: "+ str(citytracker["New York"]))

else:

    print("Sorry, that is not in the City Tracker")



if "Atlanta" in citytracker:

    print("Atlanta: "+ str(citytracker["Atlanta"]))

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for x in potentialcities:

    if x in citytracker:

        print(str(x)+ ":"+ str(citytracker[x]))

    else:

        print ("0")
for x,y in citytracker.items():

    print(x+","+ str(y))
import os



f = open("popreport.csv", "w")

f.write("city"+ ","+"pop"+"\n")

for x,y in citytracker.items():

    f.write(str(x)+ ","+ str(y)+ "\n")



f.close()    



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
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

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

key1 = user_secrets.get_secret("openCageKey") # make sure this matches the Label of your key





from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

for x in citytracker:

    citydata={} # create a new dictionary to store city data

    results = geocoder.geocode(x)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    #print ("Lat: %s, Lon: %s" % (lat, lng))

    #key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+user_secrets.get_secret("darkSkyKey")+"/"+lat+","+lng

    data = json.load(safeGet(url))

    current_time = datetime.datetime.now() 

    #citydata["time"]=current_time

    citydata["summary"]=data['currently']['summary'] # create a key summary and store summary data in city data

    citydata["temperature"]=data['currently']['temperature'] # create a key for temperature and store temperature data

    citydata["minutely"]=data['minutely']['summary'] 

    

    citytracker[x]=citydata # assign each city data to the city in citytracker dictionary 

    

    



print (citytracker)
with open("citytracker.json", 'w') as outfile: #opens a new json file and writes it

    json.dump(citytracker, outfile)