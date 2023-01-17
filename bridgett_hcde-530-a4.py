citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



#To print the population of Chicago, I use the print command to access the "Chicago" key and print the data from the citytracker dictionary.  



print(citytracker["Chicago"])
#In order to add the value 17,500 to the current value of Seattle in the dictionary, I assign a new value to the Seattle key where add 17,500. The new key should be equal to the old value plus 17,500. 

citytracker["Seattle"] = citytracker["Seattle"] + 17500



#Then I print the new population value for Seattle.

print(citytracker["Seattle"])
#Here I add the new city, Los Angeles by inserting it in the citytracker dictionary and I assign it the value of 45000. 

citytracker["Los Angeles"] = 45000



#To make sure it worked, I print the new key by calling it from the citytracker dictionary.

print(citytracker["Los Angeles"])
#First I define x by assigning the variable x the value of Denver from the citytracker dictionary.

x = citytracker["Denver"]



#to print the string,I used the +str operation. Since x has the value of Denver's population from the dictionary, I just had to print the string and add some text before it.

print("Denver's poplation is " + str(x))
#I start by using a for loop, where each city key in citytracker is iterated in a seperate line. 

for city in citytracker:

    #Then i just print the city key

    print(city)
#I start with a for loop to iterate city keys in the dictionary citytracker.

for city in citytracker:

    

    #I create the variable "population" and assign it the value of each city's population number in the citytracker dictionary.

    population = citytracker[city]

    

    #then I print the city, add colon and space text, and the string with the population values. 

    print(city + ": " + str(population))
#First I start with the New York test. I create an if statement to check if New York is in the citytracker dictionary. 

    if "New York" in citytracker:

       

    #If new york is in the dictionary, the function will print some text, plus the value of the population for that city. I used the string input the value of the city key. 

        print("New York's population is " + str(citytracker["New York"]))

    

    #If new york is not in the dictionary, the function will print an alternate sorry statement. 

    else: print("Sorry, that is not in the City Tracker.")

        

    #I do the same thing for the Atlanta test. 

    if "Atlanta" in citytracker:

        print("Atlanta's population is " + str(citytracker["Atlanta"]))

        

##At first, I tried using str(population), since I have already defined it previously, but it kept giving me the error "Los Angeles". I tried to find the error in the previous step but I don't know why I couldn't reuse it since it wasn't specific to Los Angeles. 
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
#first i create a for loop to iterate the cities in the potentialcities dictionary

for pc in potentialcities:

   

    #in each iteration, if the city in potentialcities is also in the citytracker dictionary, it will print text plus the key value.

    if pc in citytracker:

        print(pc + "'s population is " + str(citytracker[pc]))

   

    #if the potential city is not in the citytracker dictionary, the function will print 0.

    else:

        print(0)
#I start with a for loop to iterate city keys in the dictionary citytracker.

for city in citytracker:

    

    #I create the variable "population" and assign it the value of each city's population number in the citytracker dictionary.

    population = citytracker[city]

    

    #then I print the city, add comma and space text, and the string with the population values. 

    print(city + ", " + str(population))
import os



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here

#First i add the csv module manually

import csv



#here I basically create and name the file that will be where the output operation is stored

w = csv.writer(open("popreport.csv", "w"))



#I use the w.writerow function to title the columns, in this case it's city and pop.

w.writerow(["city","pop"])



#then i use a for loop to iterate the keys and their values in the citytracker dictionary

for key, val in citytracker.items():

    w.writerow([key, val])

    

#For some reason, the popreport file shows the 2 headers, but I don't get any data values in the columns and I don't know what I'm doing wrong. I'm not getting any kind of error message.



#Nevermind, I forgot I refreshed the file but didn't run the previous steps before doing this one. All good now. 

import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("scarymovie") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("scaryweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = 'Miami'  # replace this city with cities from the names in your citytracker dictionary

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



def getForecast(city="Miami"):

    key = secret_value_1

    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

    print(url)

    return safeGet(url)



data = json.load(getForecast())

print(data)

current_time = datetime.datetime.now() 



print(f"The current weather in Miami is: {data['weather'][0]['description']}")

print("Retrieved at: %s" %current_time)



### You can add your own code here for Steps 2 and 3



#To seperate the lines of code, I'm adding a sort of random division line.

print("_____________________________________________________________________") 



#I'm creating a new and empty dictionary to start with. 

city_weather = {}



#here i create a for loop to iterate the cities in the citytracker dictionary. 

for city in citytracker.keys():

    #I add this line again for division.

    print("_________________________________________________________________")

    #then the below operation defines the results of each city in the dictionary into latitude and longitude cooordinates.

    results = geocoder.geocode(city)

    #again i use these for the langitudinal and latitudinal values. 

    lat = str(results[0]["geometry"]["lat"])

    lng = str(results[0]["geometry"]["lng"])



    #next I print the results from the above. 

    #The first print function calls in each iteration of cities from the citytracker dictionary.

    print(f"{city} is located at:")

    #the second print function inputs the corresponding latitude and longitude for each city. 

    print(f"Latitude: {lat}, Longitude: {lng}")

   

    

    #next I get the weather information for each city. 

    #I wasn't sure where to input the json data - but I think putting it here is probably ok.

    city_weather[city] = json.load(getForecast(city))

    forecast = str(results[0]["temp"]["description"])

   

      #I'm having a hard time calling in and formatting the information from the API. 

    print("The weather today is " + {temp}, "with " + {description})

    

  

    

    #to put the information into a json, I open a file and give the output file a name.

    f = open("city_weather.json", "w")

    #I use this function to convert the city_weather dictionary to a json file.

    f.write(json.dumps(city_weather))

    