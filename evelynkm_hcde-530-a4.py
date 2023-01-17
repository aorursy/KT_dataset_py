citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Chicago']
citytracker['Seattle'] = 724725 + 17500

citytracker['Seattle']
citytracker['Los Angeles'] = 45000

citytracker['Los Angeles']
print("Denver:" + str(citytracker['Denver']))
for key in citytracker:

    print(key)
for key,value in citytracker.items():

    print(str(key) + ": " + str(value))
if 'New York' in citytracker:

    print("New York: " + str(citytracker['New York']))

else:

    print("Sorry, that is not in the City Tracker.")

    

if 'Atlanta' in citytracker:

    print("Atlanta: " + str(citytracker['Atlanta']))

else:

    print("Sorry, that is not in the City Tracker.")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for key in potentialcities:

    if key in citytracker:

        print(str(key) + ": " + str(value))

    else: 

        print("0")
for key,value in citytracker.items():

    print(str(key) + "," + str(value))
import os

import csv





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
#making a new dictionary to store the results of getforecast for each city in my dictionary.

new_dict_forecast = {}



#calling the APIs and using a For loop to get latitude and longitudes for each city in our citytracker list.



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage_key") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

for city in citytracker:

    query = city

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])



    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()

    secret_value_0 = user_secrets.get_secret("darksky_key") 



    import urllib.error, urllib.parse, urllib.request, json, datetime



#making sure the code continues if there's an error

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



    # lat and lon are taken from before, now being used to get the forecast for each city.

    def getForecast(lat,lng): # values taken from for loop earlier

        # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)

#printing forecast data

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    print(city + ":")

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])



#taking new empty dictionary that I started at beginning and adding all the forecast data I just printed. 

    new_dict_forecast[city] = {'Temperature': str(data['currently']['temperature']), 

                               'Time': "Retrieved at: %s" %current_time, 

                               'Current': data['currently']['summary'],

                               'Minute': data['minutely']['summary'] }



#printing new dictionary to make sure I included everything

print(new_dict_forecast)



#Now I'm saving the new dictionary as a JSON file.



import json



with open('new_dict_forecast.json', 'w') as fp:

    json.dump(new_dict_forecast, fp)