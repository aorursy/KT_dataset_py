citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker["Chicago"])
citytracker["Seattle"] += 17500 #update Seattle residency by 17,500

print(citytracker["Seattle"])
citytracker["Los Angeles"] = 45000 #insert Los Angeles into dictionary

print(citytracker["Los Angeles"]) #verify that it got added
denver_string = "Denver: "+ str(citytracker["Denver"]) #cast value associated with "Denver" key to string

print(denver_string)
for city in citytracker:

    print(city)
for city in citytracker:

    print(f"{city} : {citytracker[city]}")
#check if New York is in citytracker

if "New York" in citytracker:

    print("New York: "+ str(citytracker["New York"]))

else:

    print("Sorry, that is not in the City Tracker")

    

#check if Atlanta is in citytracker

if "Atlanta" in citytracker:

    print("Atlanta: "+ str(citytracker["Atlanta"]))

else:

    print("Sorry, that is not in the City Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for city in potentialcities:

    if city in citytracker:

        print(city+" : "+citytracker[city]) #if city in dictionary, print city and population

    else:

        print(0) #if city not in dictionary, print 0
for city in citytracker:

    print(f"{city}, {citytracker[city]}")
import os



### Add your code here



#create a file

f = open("/kaggle/working/popreport.csv","w")



#write headers

f.write("city, pop\n")



#write data

for city in citytracker:

    f.write(f"{city}, {citytracker[city]}\n")



#close file

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

query = 'Atlanta'  # replace this city with cities from the names in your citytracker dictionary

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



def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)





#Get OpenCage key

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

key1 = user_secrets.get_secret("openCageKey")

key2 = user_secrets.get_secret("darkSkyKey")





# Get DarkSky key

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_2 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key

key2=secret_value_2





#Set up OpenCage

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)





#Set up dictionary to hold all data

cityforecast ={}





#Get lat/long for each city in citytracker

#Print it out

#Save to a dictionary

for city in citytracker:

    query = str(city)

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print(city)

    print ("Lat: %s, Lon: %s" % (lat, lng))

    

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])

    print('\n')

    

    #store values as a dictionary within the dictionary of cities

    cityforecast[city]={} 

    cityforecast[city]['lat']=lat

    cityforecast[city]['lng']=lng

    cityforecast[city]['time']=str(current_time) #cast to string for JSON formatting to work

    cityforecast[city]['current']=data['currently']['summary']

    cityforecast[city]['temp']=data['currently']['temperature']

    cityforecast[city]['forecast']=data['minutely']['summary']





#print resulting dictionary holding JSON data to validate it's working

print(cityforecast)



#Save JSON output as file

f = open("/kaggle/working/cityforecast.json","w") #create a file

json.dump(cityforecast, f) #write data

f.close() #close file
