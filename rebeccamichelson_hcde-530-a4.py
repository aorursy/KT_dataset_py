citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#Calculate the number of residents in Chicago

print(citytracker['Chicago'])
#Add and print out the value of dict key 'Seattle' with its addition of 17,500 people who moved in the Summer of 2018

citytracker['Seattle'] + 17500



#Sum up 'Los Angeles', along with its new value, to the dictionary 'citytracker' and print the updated value

citytracker['Los Angeles'] = 45000

print(citytracker['Los Angeles'])
#Output both a string and value from 'citytracker', once the value is converted to a string

print("Denver: "+str(citytracker["Denver"]))
#Print each city name in 'citytracker' through iteration

for city in citytracker:

    print(city)
#Iterate through the dictionary "citytracker". Convert the integer value of "x" into a string and concatonate variable "x", ":".

for city in citytracker:

    print(city + ': ' + str(citytracker[city]))
#Check if "New York" is in the "citytracker" dictionary. If New York is in the "city tracker" dictionary, print the concatenation of the string of the integer value associated with key "New York" in dictoinary "citytracker" and this text string. If it is false, print "Sorry, that is not in the City Tracker."

if "New York" in citytracker: 

    print("New York: "+str(citytracker["New York"])) 

else:

    print("Sorry, that is not in the City Tracker")



#Check if "Atlanta" is in the "citytracker" dictionary. If Atlanta is in the "city tracker" dictionary print the concatenation of the string of the integer value associated with key "Atlanta" in dictoinary "citytracker" and this text string. If false, print, "Sorry, that is not in the City Tracker."

if "Atlanta" in citytracker:

    print("Atlanta:" +str(citytracker["Atlanta"]))

else:

    print("Sorry, that is not in the City Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

#Iterate through "city" in the list "potentialcities". If "city" is in the dictionary "citytracker" then concatenate city, ": ", and string of the integer value associated with key variable in dictionary "citytracker". If city is not in the dictionary, print zero.

for city in potentialcities:

    if city in citytracker:

        print(city + ': ' + str(citytracker[city]))

else:

    print('0')
#Iterate through the dictionary "citytracker". Convert the integer value of "x" into a string and concatonate variable "x", ":". Seperate by commas.

for city in citytracker:

    print(city + ',' + str(citytracker[city]))
import os



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here

#write a file named popreport through writer method within csv module and name it as var f

import csv

f = csv.writer(open("popreport.csv", "w"))

#Write rows ofcomma separated values

for key, val in citytracker.items():

    f.writerow((key, val))

import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("openweathermap") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print (f"{query} is located at:")

print (f"Lat: {lat}, Lon: {lng}")





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



def getForecast(city="Seattle"):

    key = secret_value_1

    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

    print(url)

    return safeGet(url)



data = json.load(getForecast())

print(data)

current_time = datetime.datetime.now() 



print(f"The current weather in Seattle is: {data['weather'][0]['description']}")

print("Retrieved at: %s" %current_time)



### You can add your own code here for Steps 2 and 3