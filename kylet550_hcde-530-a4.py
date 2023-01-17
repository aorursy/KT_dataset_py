citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



#Printing the value for the Chicago key in a string

print("There are " + str(citytracker['Chicago']) + " residents in Chicago.")
# Incrementing the Seattle value by 17,500

citytracker['Seattle'] = citytracker['Seattle'] + 17500



#Printing the incremented number of residents in Seattle

print("There are " + str(citytracker['Seattle']) + " residents in Seattle.")
#Adding Los Angeles to the dictionary and assigning a value for the Los Angeles key

citytracker['Los Angeles'] = 45000



#Printing the value associated with the Los Angeles key

print(citytracker['Los Angeles'])
#Printing a string that says Denver: X where X is the number of residents from the dictionary.

print("Denver: " + str(citytracker['Denver']))
#Iterating through citytracker and printing out each city on separate lines

for city in citytracker:

    print(city)
#Iterating through citytracker and printing out the key-value pair for each city where the key is the city name 

#and the value is the number of residents

for city in citytracker:

    print(city + " : " + str(citytracker[city]))
#Creating a function to determine whether a city is in the citytracker dictionary

def membershipTest(city):

    #Evaluating whether the input is within the citytracker dictionary

    if city in citytracker:

        

        #If the city is within citytracker, printing the city name and current population

        print(city + ": " + str(citytracker[city]))

        

    #If the city is not within citytracker, printing a message

    else:

        print("Sorry, that is not in the City Tracker.")



#Testing the function twice, once for New York and once for Atlanta

membershipTest('New York')

membershipTest('Atlanta')

            

        
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



#Creating a reusable fucntion for printing a city name (key) and population (value)

#Includes a default value of 0 for population

def printCity(city, population=0):

    print(city + ": " + str(population))

    



#Iterating through the list of cities in potentialcities to determine if they exist in citytracker

for city in potentialcities:

    

    #If the city is in citytracker, print the city name and the population

    if city in citytracker:

        printCity(city, citytracker[city])

        

    #If the city is not in citytracker, print the city name and defauled value of 0

    else:

        printCity(city)
#Prints out each individual city in citytracker as comma separated valued on different lines

#Iterates through citytracker

for city in citytracker:

    

    #Prints comma separated values of the key and value

    print(city + "," + str(citytracker[city]))
import os



### Add your code here

#Opens a file with write access

with open('popreport.csv', 'w') as outfile:

    

    #Writes the city and pop as headers for each column

    outfile.write("city" + "," + "pop" + "\n")

    

    #Itereates through citytracker and creates comma separated values for city name (key) and population (value)

    for city in citytracker:

        outfile.write(city + "," + str(citytracker[city]) + "\n")





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencagecode") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



#Create a reusable function for getting latitude and longitude

def getLatLng(city):

    results = geocoder.geocode(city)

    return results



    #Code used for testing that the application works

    #lat = str(results[0]['geometry']['lat'])

    #lng = str(results[0]['geometry']['lng'])

    #print (city + " " + "Lat: %s, Lon: %s" % (lat, lng))



#Code used for testing that the applications works

#query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

#results = geocoder.geocode(query)

#lat = str(results[0]['geometry']['lat'])

#lng = str(results[0]['geometry']['lng'])

#print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkskycode") # make sure this matches the Label of your key



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

#def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW



#Creates a reusable function for getting forecast for any latitude and longitude

def getForecast(lat, lng):

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



#Creating a new dictionary for storing data

myDiction = {}



#Iterating through each city in citytracker to get latitude and longitude

for city in citytracker:

    results = getLatLng(city)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    

    #For testing purposes, to make sure code is working

    print (city + " " + "Lat: %s, Lon: %s" % (lat, lng))

    

    #Getting forecast data for each city using reusable function for getting latitude and longitude

    data = json.load(getForecast(lat,lng))

    #current_time = datetime.datetime.now() 

    

    #Putting data for each city in dictionary

    myDiction[city] = data



#Creating a new JSON file with write access

with open('forecast.json', 'w') as outfile:

    

    #Iterating through dictionary to add city name and forecast data to JSON file

    for city in myDiction:

        outfile.write("'" + city + "': '" + str(myDiction[city]) + "\n")



#Code for testing purposes

#print(myDiction)



    #print("Retrieved at: %s" %current_time)

    #print(data['currently']['summary'])

    #print("Temperature: " + str(data['currently']['temperature']))

    #print(data['minutely']['summary'])
