citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker['Chicago']) #print the value of chicago in citytracker dictionary 
citytracker['Seattle'] = 724725 + 17500 # update Seattle key by adding 17500 to old seattle value



print(citytracker['Seattle']) #print to confirm value has changed
citytracker['Los Angeles'] = '45000'#define the Los angeles key and values and add them to citytracker dict. 

print(citytracker['Los Angeles'])#print new key value to confirm it has been added to dictionary 

citytracker['Denver'] = str(citytracker['Denver']) #cast denver population to a string. 

# Since Denver's value is not a string, 

# we need to turn it into a string so we can add it to another string for this porblem.



print ( "Denver" + ":" + citytracker['Denver']) #add Denver: to string that results in denver population and print
for city in citytracker.keys(): #iterate through citytracker dictionary keys

    print (city)#print each key

    
for city in citytracker: #iterate through the dictionary 

    population = citytracker[city] #define key value (population) as the output of the variable city in the dictionary

    print (city, " : ", population) #after each iteration, print variable city and it's corresponding population 


for city in citytracker:

    population = citytracker[city] #define key value (population) as the output of the variable city in the dictionary

    if city == "New York": #check if new york is listed as a key

        print (city, ":", population) # if it is print the city and population

    if city == "Atlanta": #check if atlanta is listed in the key

        print (city, ":", population) # if it is print the city and population 

    else:

        print("Sorry, that is not in the City Tracker") #if city is not present print the string

    
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in citytracker: #iterate through citytracker dictionary and assign each key name in dictionary to varible city

    population = citytracker[city] #define key value (population) as the output of the variable city in the dictionary

    if city in potentialcities: #check to see if city key is present in potentialcities list

        print (city, ":", population) #if it is, then pring out city name and population

    else:

        print(0) #otherwise print 0
for city in citytracker: #iterate through the dictionary 

    population = citytracker[city] #define key value (population) as the output of the variable city in the dictionary

    print (city,",",population,sep ="") #after each iteration, print variable city and it's corresponding population

    #use the sep = command to make sure items are separated without spaces

    

#another way to try this

for key, val in citytracker.items(): #set up for loop an iterate through both keys and values

    print ([key, val]) #print out each key and value
import os

import csv



w = csv.writer(open("popreport.csv", "w")) #write file



for city, pop in citytracker.items(): #set up for loop to iterate through the dictionary 

    w.writerow([city, pop]) #assign headers to each column in the dictionary and write them to the csv



### This will print out the list of files in your/working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# step one:  get things working



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey")

secret_value_1 = user_secrets.get_secret("openCageKey")

key1 = secret_value_1



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'London'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
#step one - finding lat and lng for all cities in city tracker 

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey")

secret_value_1 = user_secrets.get_secret("openCageKey")

key1 = secret_value_1



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



for city in citytracker:

    results = geocoder.geocode(city)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (city,"Lat: %s, Lon: %s" % (lat, lng))
#step 2

#get information for all of the cities in your `citytracker` dictionary. 

#You can print the information out to make sure it is working. 

#Store the results of `getForecast` for each city in your dictionary.



# This code retrieves the api keys from the Kaggle Secret Keys file and lables them for use 

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey")

secret_value_1 = user_secrets.get_secret("openCageKey")

key1 = secret_value_1



#import libraries needed for this code

import urllib.error, urllib.parse, urllib.request, json, datetime 

from opencage.geocoder import OpenCageGeocode

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



#error tracking code to make sure url retrieval is working

def safeGet(url): 

    try:

        return urllib.request.urlopen(url) #open url

    except urllib2.error.URLError as e: #if something goes wrong, return an error 

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'): #or give another type of error code

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None #if everything is fine, don't do anything 

    

for city in citytracker: #iterate through city tracker dictionary to get lat and lng of cities in dict

    #define variables

    results = geocoder.geocode(city) #results are pulled from the geocode api

    #define lat an lng as results from the geocode info

    #cast lat and lng to strings to make them easier to work with 

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])



    # define the getForecast function to look at forecasts when given a lat and lng

    def getForecast(lat,lng):

        key2 = secret_value_0 #assign key2 to secret value 0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng #call on url 

        return safeGet(url) #return results of lat and lng from the url        

    

    data = json.load(getForecast(lat,lng)) #define data variable as a json file with lat and lng

    current_time = datetime.datetime.now()  #update current date and time 

    print(city,"Retrieved at: %s" %current_time) #print current time of retrieved forecast

    print(data['currently']['summary']) #print current forecast

    print("Temperature: " + str(data['currently']['temperature'])) #print temp

    print(data['minutely']['summary']) #print projected forecast summary 

    

#step 3 Save the results of your work as a JSON formatted output file in your Kaggle output folder and Commit your notebook.

   

with open ('forecast.text', 'w') as outfile:

    json.dump (data,outfile)