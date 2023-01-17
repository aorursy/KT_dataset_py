citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print("Population of Chicago is", citytracker['Chicago'])
citytracker['Seattle']+=17500

print("Population of Seattle is", citytracker['Seattle'])
citytracker["Los Angeles"]=45000

print("Population of Los Angeles is", citytracker['Los Angeles'])
theString='Denver: '+str(citytracker['Denver'])

print(theString)
for city in citytracker.keys():

    print(city)
for city in citytracker.keys():

    print(city+":", citytracker[city])
#I'll write a small neat function, which is going to take city name and dictionary (I don't like global variables) and print what the assignment asks us to print

def cityPop(city, tracker):

    #alternatively, get() method could be used with some default value, but I found using "in" a bit more concise in this scenario

    if city in tracker.keys():

        print(city+":", tracker[city])

    else:

        print("Sorry, that is not in the Coty Tracker")



cityPop('New York',citytracker)

cityPop('Atlanta', citytracker)

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for pot in potentialcities:

    print(pot+":",citytracker.get(pot, 0))
for city in citytracker.keys():

    print(city+",", citytracker[city])
import os



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here

#I assume we need to write to the file "manually", not using external module

f=open('popreport.csv', 'w')

#writing header, including new line character

f.write("city, pop\n")

#writing each of dictionary key:value pairs, comma separated, new line character at the end

for city in citytracker.keys():

    f.write(city+", "+str(citytracker[city])+"\n")

#closing the file

f.close()

#adding package installation here, so that everything runs in one pass

!pip install opencage




import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("weather") #replace "openweathermap" with the key name you created!



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

#I will merge 2 and 3 into one loop

weather = {} #empty dictionary to start with

print("-=Weather in cities around the world=-")

for city in citytracker.keys():    

    results = geocoder.geocode(city) #getting coordinates

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{city} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    current_time = datetime.datetime.now()

    weather[city]=json.load(getForecast(city))

    print ("Weather conditions in", city,"as of",current_time,":")

    print(weather[city]['weather'][0]['description'])

    print("----------------------")

#now let's put weather into a json file

f = open('weather_in_cities.json','w') #open the file for writing

f.write(json.dumps(weather)) #convert weather to json and write into file

f.close() #close the file

print ("The weather forecast has been saved")