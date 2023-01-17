citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago'])
citytracker['Seattle'] += 17500



print(citytracker['Seattle'])
citytracker.update({'Los Angeles':45000})



#print (citytracker)

print(citytracker['Los Angeles'])
city_key = 'Denver'

print(city_key + ":" + str(citytracker[city_key]))
for key in citytracker:

    print(key)
for key in citytracker:

    print(key + ":" + str(citytracker[key]))
city_check = 'New York'

for key in citytracker:

    if key == city_check:

        print(key + ":" + str(citytracker[key]))

        break

else :

     print("Sorry, that is not in the Coty Tracker.")

        

#Same can be used to check Atlanta
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

city_check = ''

for city in potentialcities:

    city_check = city

    for key in citytracker:

        if key == city_check:

            print(key + ":" + str(citytracker[key]))

            break

    else :

         print(city_check + ": 0")
for key in citytracker:

    print(key + "," + str(citytracker[key]))
import os



### Add your code here

print("city,pop")

for key in citytracker:

    print(key + "," + str(citytracker[key]))





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

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



# retrieves forcast information for specified latitude and longitude

#lat and lon below are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



#retrieves and stores city information dictionary

for city in citytracker:

    #retrive city coordinates

    results = geocoder.geocode(city)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    #use city coordinates to get forcast information

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    #create and store new dict of information in citytracker

    citytracker[city] = []

    citytracker[city].append({

        "latitude": lat,

        "longitude": lng,

        "time": str(current_time),

        "current_summary": str(data['currently']['summary']),

        "temp": str(data['currently']['temperature']),

        "minute_summary": data['minutely']['summary']

    })



#write city information into json file

with open('citydata.txt', 'w') as outfile:

    json.dump(citytracker, outfile)



#print city information in pretty json formating

city_json = json.dumps(citytracker, indent=4, sort_keys=True)

print(city_json)


