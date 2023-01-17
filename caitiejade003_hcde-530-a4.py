citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



print(citytracker['Chicago'])
citytracker['Seattle'] = citytracker['Seattle'] + 17500 #replacing the current value for the key seattle with the value + 17500

print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000 #add a new item

print(citytracker['Los Angeles']) #printing the value
x = citytracker['Denver'] #Creating a variable to hold the value

print("Denver: " + str(x)) #print the value as a string
for item in citytracker.keys(): #for loop looking at each item in the dictionary

    print(item)
for city in citytracker.keys(): #assign city as the key and iterate through each one

    print (city + ": " + str(citytracker[city])) #print the key plus the value
sublist = ['New York', 'Atlanta'] #creating list of the two cities we want to check



for item in sublist: #iterate through the list

    if item in citytracker.keys():  #check if that item is a key in our dictionary

        print(item + ": " + str(citytracker[item])) #if so, print out the key followed by it's value

    else:

        print("Sorry, that is not in the City Tracker")  #otherwise print the "sorry" message, changed Coty to City
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



def city_checker(city):  #creates function that takes parameters of city name

    pop = citytracker.get(city, 0) #makes a variable to hold the value of each a key with a default of 0 if the key isn't found

    print(city + ": " + str(pop))  #prints the city name and the value if the key exists, otherwise shows city name and default value



for item in potentialcities:    #looping through the list

    city_checker(item)         #placing each element of the list as the parameter of the function



for city in citytracker.keys(): #assign city as the key and iterate through each one

    print (city + "," + str(citytracker[city])) #print the key plus the value with comma and no spaces
import os



### Add your code here

import csv



with open('popreport.csv', mode='w') as popreport_file:  #open a new file in write mode and assign it to popreport_file

    pop_writer = csv.writer(popreport_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #assign parameters to the csv writer method, found this online so not sure what the purpose quotechar/quoting parts are

    pop_writer.writerow(['city', 'pop'])             #write the first row to be a header

    for city in citytracker.keys():                                     #iterate through city tracker keys

        pop_writer.writerow([city, citytracker[city]])                  #use writer and writerow method to add each key and it's value

    





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("Open Cage") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

for city in citytracker.keys():                                     #iterate through city tracker keys

    query = city  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    citytracker[city] = coordinates = {'lat' : lat, 'long' : lng}  #updating each city in citytracker with a sub dictionary of the coordinates

    print (city + ": " + "Lat: %s, Lon: %s" % (lat, lng)) #checking my work

    

print(citytracker['Los Angeles']['lat']) #checking my work

    





#saving original code in case I need it

#from kaggle_secrets import UserSecretsClient

#user_secrets = UserSecretsClient()

#secret_value_0 = user_secrets.get_secret("Open Cage") # make sure this matches the Label of your key

#key1 = secret_value_0



#from opencage.geocoder import OpenCageGeocode

#geocoder = OpenCageGeocode(key1)

#query = 'Los Angeles'  # replace this city with cities from the names in your citytracker dictionary

#results = geocoder.geocode(query)

#lat = str(results[0]['geometry']['lat'])

#lng = str(results[0]['geometry']['lng'])

#print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("Dark Sky") # make sure this matches the Label of your key



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



#variable to store 



# lat and lon below are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)







for cityname in citytracker:  #go through each first level of the dictionary

    lat = citytracker[cityname]['lat']  #redefine the lat variable by getting that element from the dictionary

    lng = citytracker[cityname]['long'] #redefine the long variable by getting that element from the dictionary



    data = json.load(getForecast(lat,lng))  #using function defined above

    current_time = datetime.datetime.now()  #for some reason this one wasn't working so need to investigate...everything else works

    #adding the forecast data as sub dictionary to each cityname

    citytracker[cityname].update({'Timestamp' : str(current_time), 'Current' : data['currently']['summary'], 'Temp' : data['currently']['temperature'], 'Future' : data['minutely']['summary']})



    #original print statements

    #print("Retrieved at: %s" %current_time)

    #print(data['currently']['summary'])

    #print("Temperature: " + str(data['currently']['temperature']))

    #print(data['minutely']['summary'])



with open('allforecasts.txt', 'w') as outfile:  #creating a new file to store the dictionary

    json.dump(citytracker, outfile)  #dumping the dictionary as json



# lat and lon below are for UW (original code)

#def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

 #   key2 = secret_value_0

 #   url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

 #   return safeGet(url)
