citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



# print the value associated with "Chicago" in the dictionary

citytracker['Chicago']
# replace the value of Seattle with a new value that adds 17,500 to the old value

citytracker['Seattle'] = citytracker['Seattle'] + 17500



# print the new value for Seattle

citytracker['Seattle']
# add a new city (Los Angeles) with a population of 45,000

citytracker['Los Angeles'] = 45000



# print the value associated with Los Angeles

citytracker['Los Angeles']
# assign the variable denverPop to the value associated with Denver in the dict.

denverPop = citytracker['Denver']



# print the string

print("Denver: %d" %denverPop)
# print all the cities in citytracker

for city in citytracker:

    print(city)
# iterate through citytracker and print each key with its number of residents

for city, population in citytracker.items():

    print(city, ":", population)
# a function that checks if the city is in citytracker

def cityCheck(city):

    

    # if city is in city tracker, print city and value

    if city in citytracker:

        print(city, ":", citytracker[city])

        

    #if city is not in city tracker, print error message

    else:

        print("Sorry, that is not in the City Tracker")



#check if New York is in city tracker

cityCheck("New York")



#check if Atlanta is in city tracker

cityCheck("Atlanta")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



# a function that checks if the city is in citytracker

def potentialCityCheck(city):

    

    # if city is in city tracker, print city and value

    if city in citytracker:

        print(city, ":", citytracker[city])

        

    #if city is not in city tracker, print 0

    else:

        print(0)



# check to see if any cities in potentialcities are in citytracker

for city in potentialcities:

    potentialCityCheck(city)
# iterate through citytracker and print each key with its number of residents

for city, population in citytracker.items():

    print(city, ",", population)
import os



### Add your code here



import csv



# create the csv file

f = open('popreport.csv', 'w')

writer = csv.writer(f)



# write the headers

writer.writerow(["city", "pop"])



# iterate through citytracker and write the key, value pairs

for key, value in citytracker.items():

    writer.writerow([key, value])



# close the file

f.close()





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage") # make sure this matches the Label of your key

key1 = secret_value_0



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSky") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



# create new dictionary called cityweather

cityweather = {}



# iterature through all cities in city tracker and get their lat longs and weather

for city in citytracker:

    from opencage.geocoder import OpenCageGeocode

    geocoder = OpenCageGeocode(key1)

    query = city # replace this city with cities from the names in your citytracker dictionary

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

    def getForecast(lat,lng):

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)



    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    

    # print the results of the API calls to verify things are working as expected

    print(city)

    print ("Lat: %s, Lon: %s" % (lat, lng))

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])

    

    # update cityweather dictionary with key as city and then the data as the value

    cityweather = {

        city : {'latitude':lat, 

                'longitude':lng, 

                'retrieved at time':str(current_time), 

                'weather':data['currently']['summary'],

                'temperature':str(data['currently']['temperature']),

                'summary':data['minutely']['summary']}

    }

    

    # update the original dictionary with the cityweather dictionary

    citytracker.update(cityweather)



    # print the dictionaries to verify

    print(cityweather)

    print(" ")

    print(citytracker)

    print("")

    

    # write the  citytracker dictionary that now has weather info to a json file

    with open('jsonfile.json', 'w') as json_file:

        json.dump(citytracker, json_file)