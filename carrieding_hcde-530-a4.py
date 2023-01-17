citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Chicago']
citytracker['Seattle'] = 724725 + 17500

citytracker['Seattle']
citytracker['Los Angeles'] = 45000

citytracker['Los Angeles']
print("Denver: " + str(citytracker['Denver']))
for city in citytracker:

    city != int #evaluate city(key) is not equal to an interger

    print(city)
for city, population in citytracker.items():

    print(city,":", population)
# for loop goes through the dictionary to evaluate if city is New York or Atlanta, print the Sorry statement if not.



for city, population in citytracker.items():

    if city == 'New York':

        print(city, ':', population)

    elif city == 'Atlanta':

        print(city, ':', population)

    else:

        print('Sorry, that is not in the Coty Tracker.')
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
# for loop check if any city in the potentialcities list is in citytracker dictionary, print 0 if not.



for city, population in citytracker.items():

    if city in potentialcities:

        print(city,":", population)

    else:

        print('0')
for city, population in citytracker.items():

    print(str(city) + ',' + str(population))
import os



### Add your code here

f = open("popreport.csv", "w") #open a file named popreport.csv and write in this file

f.write("city" + "," + "pop" + "\n") #write city name, population and end with a new line

for city, population in citytracker.items(): #for loop goes through the citytracker dictionary

    f.write(str(city) + ',' + str(population) + '\n') #record the data in the format of city name, population, add a new line



f.close()



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# step 1

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("OpenCageKey")

# make sure this matches the Label of your key

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

secret_value_0 = user_secrets.get_secret("DarkSkyKey")

secret_value_1 = user_secrets.get_secret("OpenCageKey") # make sure this matches the Label of your key



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

# Step 2



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSkyKey")

secret_value_1 = user_secrets.get_secret("OpenCageKey")

key1 = secret_value_1 #assign a variable name to unique API key for OpenCage



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

for city in citytracker: #assign a variable name to key in citytracker dictionary

    citylist = {} #create a new list named citylist to store the key, city

    print(city)

    results = geocoder.geocode(city)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print ("Lat: %s, Lon: %s" % (lat, lng)) #print latitude and longtitude of a specific city

    key2 = secret_value_0 #assign a variable name to unique API key for DarkSky

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng #the web address to access API using my unique key

#    print(url)

    data = json.load(safeGet(url))

    

    current_time = datetime.datetime.now() 

    #citylist["time"] = current_time

    citylist["summary"] = data['currently']['summary']

    citylist["Temperature"] = data['currently']['temperature']

    citylist["minute"] = data['minutely']['summary']

    

    #store the above data of the cities in citytracker in a citylist dictionary.

    #In other words, citylist is a sub dictionary hiding within citytracker dictionary.

    citytracker[city] = citylist 

    

print(citytracker)
# Step 3



#open a json-formatted file named cityforecast to store as output

#write the data in citytracker in this file

with open("cityforecast.json", "w") as outfile:

    json.dump(citytracker, outfile)