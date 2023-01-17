citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker["Chicago"]) #prints the value of the key
Seattle_Population = citytracker["Seattle"]



citytracker["Seattle"] = Seattle_Population + 17500 #adds 17500 to the population of seattle



print(citytracker["Seattle"]) #prints the value of the key
citytracker["Los Angeles"] = 45000 #adds a new key value pair



print(citytracker["Los Angeles"])
def city_spitter(x):

    for city in citytracker.keys(): #iterates through the keys 

        if x in city: #checks to see if the entered city is in the dictionary

            return x + " : " + str(citytracker[x]) #returns the population and the city

            

    return "sorry, that city isn't in the database" #returns an oopsie when the city doesn't exist 



print(city_spitter("Denver"))
for city in citytracker.keys(): #iterates through the city values

    print(city) #prints each value
for city in citytracker.keys():

    print(city + ": " + str(citytracker[city])) #same as above but also prints the pop

    
def city_spitter(x):

    for city in citytracker.keys(): #works for any city! 

        if x in city:

            return x + " : " + str(citytracker[x])

            

    return "sorry, that city isn't in the database"



print(city_spitter("New York"))



#My city spitter function does this already comments same as above.
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee', "Seattle"]



def city_spitter_two():

    for city in potentialcities: #loops through potential cities 

        if city in citytracker: #checks to see if the city is in city tracker

            return city + " : " + str(citytracker[city]) 

            

    return " 0"



print(city_spitter_two()) #not sure if this is what this assignment is asking for or not; not 100% clear but I added Seattle to potential cities to test if it woudl catch and it did? 
for city in citytracker.keys():

    print(city + "," + str(citytracker[city])) #prints city and a comma and the pop
print(type(citytracker))

import os

import csv



### Add your code here

csv_file = "popreport3.csv"

with open(csv_file, "w") as file: 

    writer = csv.writer(file) #uses the fancy csv library to make a shortcut variable (?)

    writer.writerow(["city", "pop"]) #uses that shortcut vvariable to start writing rows, this is for the header

    for key, value in citytracker.items(): # loop that loops through and prints my key values to the csv file. 

        writer.writerow([key, value])





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage") # make sure this matches the Label of your key

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
