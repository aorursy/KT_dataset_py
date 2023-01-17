citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(citytracker['Chicago'])#Search for the value of Chicago in the dictionary "citytracker"
citytracker['Seattle']=(citytracker['Seattle']+17500)#Change the value of key 'Seattle' by adding 17500 to the original value

print(citytracker['Seattle'])#Print new value
citytracker['Los Angeles']=45000#Adds a new key:value pair to the dictionary

print(citytracker['Los Angeles'])
print('Denver:%d' %citytracker['Denver'])

#Signal to python with the % that the string will call an integer

#Then define the integer with a second % by calling the value in the dictionary
cities=citytracker.keys()#Define variable for a list of just the keys from the citytracker dictionary

for citynames in cities:#Loop that variable of keys

    print(citynames)
for citynames in cities:

    print(citynames,": %d" %citytracker[citynames])

#Uses the % to call the value of the key input by the loop "citynames"
if 'New York' in citytracker: #Searches for value New York in city tracker dictionary

    print('New York : %d' %citytracker['New York'])#If true, will print New York and its value

else:

    print("Sorry that is not in the City Tracker")#If false, will print message informing that it's not in the dictionary



#Same formula below with input for "Atlanta"

if 'Atlanta' in citytracker:

    print('Atlanta : %d' %citytracker['Atlanta'])

else:

    print("Sorry that is not in the City Tracker")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for search in potentialcities:#Loops the new list

    if search in citytracker:#Checks if strings in the new for loop exist in dictionary citytracker

        print(search,' : %d' %citytracker[search])#If true, prints city:population

    else:

        print(search, ": 0")#Otherwise, prints zero
for citynames in cities:#Same loop as question 6

    print(citynames+",%d" %citytracker[citynames])#Slight change in code, using a + instead of , between the loop and %integer, so that there will not be a space in the print

    #Also replaced the ":" with a ","
import os



### Add your code here

import csv

with open('popreport.csv','w', newline='') as outfile:

    thewriter=csv.DictWriter(outfile, fieldnames=['City','Pop'])#Assigns a variable to the writer that I am using to creat the CSV

    thewriter.writeheader()

    for citynames in cities:

        thewriter.writerow({'City':citynames, 'Pop':citytracker[citynames]})



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage") # make sure this matches the Label of your key

key1 = secret_value_0

#from opencage.geocoder import OpenCageGeocode

#geocoder = OpenCageGeocode(key1)

#query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

#results = geocoder.geocode(query)

#lat = str(results[0]['geometry']['lat'])

#lng = str(results[0]['geometry']['lng'])

#print ("Lat: %s, Lon: %s" % (lat, lng))

#print(cities)

places={}

def latlong(x):#Create a procedure to find the lattitudes and longitudes for each city

    from opencage.geocoder import OpenCageGeocode

    geocoder = OpenCageGeocode(key1)

    query = (x)  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    places[x]=lat,lng#added a key here to store all of the values discovered by this function in a dictionary

    print ('\'%s\''%lat+','+'\'%s\''%lng)







for locations in cities:#Creating a loop for the keys only

    latlong(locations)#using my latitude longitude finder to discover all of the lats/lngs for these cities in my dictionary

    print (locations)
zoomed=list(places.values()) 

zoomed

#Here is where I got stuck. I can get a list of all the values in the dictionary which have been assigned to 

#the key of their locations. However, the values all have a () around them, so when I 

#insert them into the function it's only reading it as one command and not reading it as two. So for example:

#getForecast(zoomed[0]) was read as getForecast(('33.7490987', '-84.3901849')) and wasn't finding a command for lng

#If I could have gotten this to work, I would have had a much better code.
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSky") # make sure this matches the Label of your key



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



#getForecast(lat="33.7490987",lng="-84.3901849")

#getForecast(lat="42.3602534",lng="-71.0582912")

#getForecast(lat="41.8755616",lng="-87.6244212")

#getForecast(lat='39.7392364',lng='-104.9848623')

# lat and lon below are for UW

def getForecast(lat,lng): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



url=getForecast('33.7490987', '-84.3901849')



data = json.load(url)

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])





url=getForecast('33.7490987', '-84.3901849')#cast a variable that will define the URL that this function returns



data = json.load(url)#defining variable data as the json script to read that URL

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)#Print cues for successful 

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])
url=getForecast('42.3602534', '-71.0582912')



data1 = json.load(url)#changing the 'data' variable a little each time

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data1['currently']['summary'])

print("Temperature: " + str(data1['currently']['temperature']))

print(data1['minutely']['summary'])
url=getForecast('41.8755616', '-87.6244212')



data2 = json.load(url)

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data2['currently']['summary'])

print("Temperature: " + str(data2['currently']['temperature']))

print(data2['minutely']['summary'])
url=getForecast('39.7392364', '-104.9848623')



data3 = json.load(url)

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data3['currently']['summary'])

print("Temperature: " + str(data3['currently']['temperature']))

print(data3['minutely']['summary'])
url=getForecast('47.6038321', '-122.3300624')



data4 = json.load(url)

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data4['currently']['summary'])

print("Temperature: " + str(data4['currently']['temperature']))

print(data4['minutely']['summary'])
url=getForecast('34.0536909', '-118.2427666')



data5 = json.load(url)

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data5['currently']['summary'])

print("Temperature: " + str(data5['currently']['temperature']))

print(data5['minutely']['summary'])
import json



ultimatedata=[data,data1,data2,data3,data4,data5]#I created a list to store all the variable data together for the json dump. 

with open("cityweather.txt", 'w') as outfile:

    json.dump(ultimatedata,outfile)