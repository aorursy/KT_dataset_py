citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

print(str(citytracker["Chicago"])+ " residents in Chicago") #cast and call the value for key "Chicago" in dictionary 'citytracker', then print this value
citytracker["Seattle"]=citytracker["Seattle"]+17500 #update the value for key "Seattle" to the existing value for key "Seattle" plus 17500

print(str(citytracker["Seattle"])+" residents in Seattle") #print updated value for key "Seattle" in dictionary "citytracker"
LA="Los Angeles" #create new variable LA, make it equal to the string "Los Angeles"

citytracker[LA]= 45000 #for the key associated with variable "LA" in the dictionary "citytracker", set the value to 45000

                #I wasn't sure whether or not to set this as "45000" or 45000 without quotes. Since the above items in the dictionary are green/integers, I did not include quotes

print(citytracker[LA])#print value associated 

print("Denver: "+str(citytracker["Denver"]))

#broken down:

#citytracker["Denver" #call the value associted with key "Denver" in dictionary "citytracker"

#str(citytracker["Denver"])) #convert the integer value to a string

#print("Denver: "+str(citytracker["Denver"])) #print the concatenation of these two strings
for x in citytracker.keys(): #create variable x; loop through the keys of dictionary "citytracker"

    print(x) #print variable x of each loop
for x in citytracker.keys(): #create variable x; loop through the keys of dictionary "citytracker"

    print(x+": "+str(citytracker[x])) #print, in one string, the concatenation of variable x, ": ", and the string conversion of the integer value associated with key x
if "New York" in citytracker: #see if this key is in dictionary "citytracker"

    print("New York: "+str(citytracker["New York"])) #if the above is true, print the concatenation of this text string and...

                                                    #the string of the integer value associated with key "New York" in dictoinary "citytracker"

else:

    print("Sorry, that is not in the City Tracker")#if false, print this string



if "Atlanta" in citytracker:#see if this key is in dictionary "citytracker"

    print("Atlanta:" +str(citytracker["Atlanta"]))#if the above is true, print the concatenation of this text string and...

    #the string of the integer value associated with key "Atlanta" in dictoinary "citytracker"

else:

    print("Sorry, that is not in the City Tracker")#if false, print this string



potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

for city in potentialcities: #iterate through the temporary variable "city" in the list "potentialcities"

    if city in citytracker: #if var "city" is in the dictionary "citytracker"

        print(city+":"+str(citytracker[city]))#print concatenation of city, ": ", and string of the integer value associated with key variable in dictionary "citytracker"

    else:

        print("0")
#below code assumes that city goes in one cell and population in another, with the colon omitted

for x in citytracker.keys(): #create variable x; loop through the keys of dictionary "citytracker"

    print(x,str(citytracker[x]), sep=",") #print, on one line, variable x and the string conversion of the associated integer value of its key

                                        #separated by a comma
import os

print(os.getcwd()) 



### Add your code here

#below code assumes that city goes in one cell and population in another, with the colon in "key:value" omitted

import csv #import the csv module

with open('popreport.csv', 'w', newline='') as f: #create the file object popreport.csv in write mode

                                                #include newline=''because the python documentation for the csv module says to do this(i don't fully understand why)

    writer = csv.writer(f) #return a writer object for the file object f

    writer.writerow(["city","pop"])#use the writewrow method on object writer, writing the string items in [] as the first row to indicate column headers

    writer.writerows(citytracker.items())#use the writerows method on object writer, printing the key:value pair on each line



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install opencage #per Rafal, adding this code so I don't have to install the module every time I come back to kaggle



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage Geocoder")

secret_value_1 = user_secrets.get_secret("OpenWeather")



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print (f"{query} is located at:")

print (f"Lat: {lat}, Lon: {lng}")



#STEP 2A

for c in citytracker: #create temporary variable c; use this temp variable to iterate through the dictionary "citytracker"

    query = c #create the object "query", make it equal the string value associated with temporary variable c

    results = geocoder.geocode(query) #create the object "results", make it equal to the return value of method geocode on object geocoder for the object query

    lat = str(results[0]['geometry']['lat']) #create the object "lat", make it equal to the string value of the integer returned by results[0]['geometry']['lat']

    lng = str(results[0]['geometry']['lng'])#create the object "lng", make it equal to the string value of the integer returned by results[0]['geometry']['lng']

    print (f"{query} is located at:") #print this string

    print (f"Lat: {lat}, Lon: {lng}") #print this string
#STEP 2B: I wrote in some comments of what I think the code provided for me did to test my own understanding



import urllib.error, urllib.parse, urllib.request, json, datetime #import these modules



def safeGet(url): #define the new function safeGet for the unique variable (url)

    try: #test this block of code for errors

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e: #handle this error depending on the conditional logic...

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



def getForecast(city="Seattle"): #define the new function getForecast for the unique variable

    key = secret_value_1 #create variable key, set it equal to my OpenWeather API key saved in Kaggle secrets add on 

    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key #define the variable url as this concatenated string

    print(url)

    return safeGet(url)



#data = json.load(getForecast()) #create new object 'data', set it equal to the return of the file object of getForecast and return it as a json object

#print(data) #print the json object data

#current_time = datetime.datetime.now() #create object current_time, set it equal to the return of this function



#print(f"The current weather in Seattle is: {data['weather'][0]['description']}") #print this string, which calls the results of the 'weather' subdictionary

#print("Retrieved at: %s" %current_time)



for d in citytracker: #create temporary variable 'd', then iterate through the dictionary 'citytracker'

    datad= json.load(getForecast(d)) #create object datad, set it equal to the return of the file object of getForecast(d) and return it as a json object

    #I found that this creates a broken URL for the city "Los Angeles" because of the space, but if I change the code in def getForecast to put city.replace(" ","")...

    #...I get an error in the defsafeGet, where urllib returns a name error

    citytracker[d]=[citytracker[d],datad] #update dictionary citytracker so each key[d] is updated with the values citytracker[d] and datad



#I printed the below to verify that the getForecast data is associated with each city in the citytracker dictionary

for x in citytracker.keys(): #create variable x; loop through the keys of dictionary "citytracker"

    print(x,str(citytracker[x]), sep=",") #print, on one line, variable x and the string conversion of the associated integer value of its key

                                        #separated by a comma



#STEP 3





with open('citytrackerwforecast.json', 'w') as outfile: #create the output .json encoded file named 'citytrackerwforecast' and open in write mode

    json.dump(citytracker, outfile) #serialize the object 'citytracker' in a json encoded stream to the outfile filepath (I'm not sure if I am using all the corret terminology)







# This will print out the list of files in my /working directory to confirm I wrote the file.

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))