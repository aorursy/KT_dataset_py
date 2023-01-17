citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print("Number of Residents in Chicago: %d" %citytracker.get('Chicago'))
citytracker['Seattle'] = citytracker['Seattle'] + 17500

print("Number of residents in Seattle: %d" %citytracker['Seattle'])
citytracker['Los Angeles'] = 45000

print("Number of residents in Los Angeles: %d" %citytracker['Los Angeles'])
print("Denver:" + str(citytracker['Denver']))
for city in citytracker.keys():

    print(city)

    
for city in citytracker.keys():

    print(city +": " + str(citytracker[city]))
if 'New York' in citytracker.keys():

    print("New York:" + str(citytracker['New York']))

else:

    print("Sorry, that is not in the City Tracker")



if 'Atlanta' in citytracker.keys():

    print("Atlanta: " + str(citytracker['Atlanta']))

else:

    print("Sorry, that is not in the City Tracker")
potential_cities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee','Atlanta']



for city in potential_cities:

    print(city + ": " + str(citytracker.get(city, 0)))
for city in citytracker.keys():

    print(city +"," + str(citytracker[city]))
import os



# Create new csv file and add headers

report = open("popreport.csv", "a")

report.write("city,pop\n")



#populate each row of csv with city name and population 

for city in citytracker.keys():

    report.write(city +"," + str(citytracker[city]) + "\n")

    

report.close()



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