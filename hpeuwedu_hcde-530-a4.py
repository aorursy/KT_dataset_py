citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#dictionary name calling key 'Chicago'

citytracker['Chicago']
#giving Seattle key a new value

citytracker["Seattle"] = 724725 + 17500 

#give the new value of "Seattle"

citytracker["Seattle"]
#adding key Los Angeles with a value of 45000

citytracker["Los Angeles"] = 45000

#print function to check if Los Angeles key was added to citytracker dictionary

print(citytracker)


citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#defining a new variable for citytracker['Denver']

#pop = citytracker['Denver']

#printing "Denver: 619968" using string function to print the value of Denver

#print("Denver:" + str(pop))

print("Denver:" + str(citytracker['Denver']))
#def variable 

#name = 0 

#for loop to iterate through citytracker dictionary

for cname in citytracker:

    #defining a variable to iterate through citytracker and to get the value of each key

    pop = citytracker[cname]

    #prints city name with a space and the population number

    print(cname + " " + str(pop))


#for loop to iterate through citytracker dictionary

for cname in citytracker:

    #defining a variable to iterate through citytracker and to get the value of each key

    pop = citytracker[cname]

    #prints city name with a colon, and the population number

    print(cname + ":" + str(pop))
#creating a function (I think its a function) for finding a city in dictionary

def findcity(citytracker,city): 

    #if statement using city as a variable to go through dictionary

    if city in citytracker:

        #if the city is is found in the dictionary print name, colon, and value

        print(city + ":" + str(citytracker[city]))

        #if not found (else)

    else:

        #print the following

        print("Sorry, that is not in the City Tracker.") 

        

        

 #defining city to look for       

city = "New York"

#i think this means that findcity function will look through dictionary called citytracker to find "New York" because

#city is referring to New York

findcity(citytracker,city)



#defining new city to look for

city = "Atlanta" 

findcity(citytracker,city)

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee'] 



#I think this is a function for finding a city in dictionary

def findcity(citytracker,city): 

    #if statement using city as a variable to go through dictionary

    if city in citytracker:

        #if the city is is found in the dictionary print name, colon, and value

        print(city + ":" + str(citytracker[city]))

    else: #if not found

        #print the following

        print(city + ":" + "0") 

              

 #defining city to look for       

city = "Cleveland"

#i think this means that findcity function will look through dictionary called citytracker to find "New York" because

#city is referring to New York

findcity(citytracker,city)

#defining new city to look for

city = "Pheonix" 

findcity(citytracker,city)

#defining new city to look for

city = "Nashville" 

findcity(citytracker,city)

#defining new city to look for

city = "Philadelphia" 

findcity(citytracker,city)

#defining new city to look for

city = "Milwaukee" 

findcity(citytracker,city)



#for loop to iterate through citytracker dictionary

for cname in citytracker:

    #defining a variable to iterate through citytracker and to get the value of each key

    pop = citytracker[cname]

    #prints city name with a comma, and the population number

    print(cname + "," + str(pop))
import os



### Add your code here 



#i think this creates the file object

with open('popreport.csv', 'w') as f:

    writer = csv.writer(f) # this is defining what we're writing in the file?

    for cname in citytracker: #for loop to iterate over the dictionary

        writer.writerow([cname, citytracker[cname]]) #this should write the city and pop in rows

f.close() #closes file because its good form. 

         

### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("open cage") # make sure this matches the Label of your key

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

secret_value_0 = user_secrets.get_secret("open cage") # make sure this matches the Label of your key

darkskykey = user_secrets.get_secret("dark sky")

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

    url = "https://api.darksky.net/forecast/"+darkskykey+"/"+lat+","+lng

    print(url)

    return safeGet(url)



data = json.load(getForecast()) #originally had lat & long in this function that was calling lat an

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])
