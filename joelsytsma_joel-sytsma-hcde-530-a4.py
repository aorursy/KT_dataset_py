citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle']=citytracker['Seattle']+17500 #I'm redefining Seattle to add 17500 to the number associated with the Seattle key



print(citytracker['Seattle'])
citytracker['Los Angeles'] = 45000 #setting a new key for the city tracker dictionary and defining it's value to 45000



print(citytracker['Los Angeles'])
print("Denver:", citytracker['Denver']) # printing the word Denver, followed by the value associated with the Denver key
for city in citytracker: #setting the city variable to inherit the key for each loop of the citytracker dictionary

    print(city) #printing the variable we've assigned the key value to
for city in citytracker: ##setting the city variable to inherit the key for each loop of the citytracker dictionary

    print(city, ":", citytracker[city]) #added code so that the variable that city currently is referencing calls that value in each iteration
if 'New York' in citytracker: #querying citytracker keys for the "New York" key

    print ('New York',":", citytracker['New York']) #if it's true then it would print this. New york isn't there, so it won't print this

else:

    print("Sorry, that is not in the Coty Tracker") #setting up and else statement to print if New York doesn't appear

    

if 'Atlanta' in citytracker: #same logic as New York, but for Atlanta

    print ('Atlanta',":", citytracker['Atlanta'])

else:

    print("Sorry, that is not in the Coty Tracker")

    
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee', 'Chicago'] #I added Chicago to the list, just to make sure my code below worked



for city in potentialcities: #looping through the potentialcities list and assigning the value in each index the variable "city"

    if city in citytracker: #creating a conditional statement to see if the index value is a key in the citytracker dictionary

        print(city,":", citytracker[city]) #if "city" is in citytracker, then print it's appropriate value

    else: #if city is not in the dictionary

        print("0") #then print 0

        
for city in citytracker: #same as problem 6

    x=citytracker[city] #setting variable x to be the value to the pair that the city key calls. Doing this because the format function doesn't accept dictionaries.

    print(city+",{}".format(x)) #printing the city variable plus ","" and the text that was placed in the x. I removed the spaces using format.
import os

import csv #importing the functionality to write a CSV



with open('popreport.csv', 'w') as f: #using the open function to write to the file popreport.cvv. I"m going to put what I want to write in the variable f.

    for city in citytracker: #looping through the citytracker dictionary to find each key value

        x=citytracker[city] #setting up the x variable to represent the value of each key that is found

        print(city+",{}\n ".format(x)) #just keeping track of each loop

        f.write(city+",{}\n".format(x)) #writing each loop to f. the format function removes spaces and \n starts a new line after each loop.

    







### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCage") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

for query in citytracker:

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (query,"Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSky") # make sure this matches the Label of your key



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



def getForecast(lat="lat",lng="lng"):

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

        key2 = secret_value_0

        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

        return safeGet(url)



cityweather = {}



for query in citytracker: #put everything in a for loop

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print(query)

    getForecast(lat,lng)

    data = json.load(getForecast(lat,lng))

    current_time = datetime.datetime.now() 

    

    #all the variables, consider using the .update

    print("Retrieved at: %s" %current_time)

    print(data['currently']['summary'])

    print("Temperature: " + str(data['currently']['temperature']))

    print(data['minutely']['summary'])

    

    cityweather[query] = {'current time':current_time, 'temperature': data['currently']['temperature'],'mini forecast':data['minutely']['summary']}

with open('jsonfile.json', 'w') as json_file:

        json.dump(citytracker, json_file)

        