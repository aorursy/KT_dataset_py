citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



city = 'Chicago' #I'll set the city to the name of the city I want to work with.

print('There are %d residents in ' %citytracker[city] + city +'.') #I'll print a message that says how many residents are in the city,
seattlebefore = 724725 #I want to store this value separately for reasons.

citytracker['Seattle'] = citytracker['Seattle'] + 17500

print(seattlebefore)

print(citytracker['Seattle'])

print('Seattle had %d and now it has %d.' %(seattlebefore,citytracker['Seattle'])) #The only problem with this is it goes higher and higher each time this code block is run within the notebook, unless you go re-run the first code block.
citytracker['Los Angeles'] = 45000 #This will add Los Angeles to the dicrionary.

print(citytracker['Los Angeles']) #This will print the value of Los Angeles in the dictionary.
print("'Denver: %d\'" %citytracker['Denver']) #I am taking the markdown very iterally. I am using this touple string formatting instead of str().
for i in citytracker.keys(): #Setup the for loop and look in keys rather than values. 

    print(i) #Use the temporary variable i.
for i in citytracker.keys(): #Setup the for loop same as above.

    j = citytracker[i] #Lookup the value associated with the key.

    print(i +" : %d" %j) #Format the output. Again I am using the touple formatting. 

#I don't know a 'Coty Tracker' but I can check in the city tracker.

cities = ['New York','Atlanta'] #I will setup a list of the cities I want to search for.

for i in cities: #Here is a for loop to step through my list.

    if i in citytracker: #Here is a conditional checking if the city is in the list.

        print(i +": %d" %citytracker[i]) #Here is the formatted output for the true condition.

    else:

        print("Sorry, that is not in the Coty Tracker.") #Here is the formatted output for the false condition.
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

#Because I setup a list and for loop above, I can reuse a lot of that code.

for i in potentialcities: #Change the thing I am iterating through.

    if i in citytracker: #Here is a conditional checking if the city is in the list.

        print(i +": %d" %citytracker[i]) #Here is the formatted output for the true condition.

    else:

        print(0) #Here is the formatted output for the false condition.
#Reusing the code from step 6 but modifying the output to use a comma and no space, rather than a colon.

for i in citytracker.keys(): #Setup the for loop same as above.

    j = citytracker[i] #Lookup the value associated with the key.

    print(i +",%d" %j) #Format the output. Again I am using the touple formatting. 



import os



### Add your code here



#I need to set a file handle for my output file. This will create the file if it doesn't exist.

out = open("popreport.csv", "w")



for i in citytracker.keys(): #Setup the for loop same as above.

    j = citytracker[i] #Lookup the value associated with the key.

    out.write(i +",%d\n" %j) #Format the output. Again I am using the touple formatting. I have added \n to create a new line after each entry.



out.close() #Close the file when I am done. 



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
!pip install opencage
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





#Now my code.

#I will basically reuse the code from above, but I will put it into a for loop and replace Seattle with the city retrieved from the for loop.

#for city in citytracker.keys(): 

#    query = city #Here is where we grab the city name as the loop iterates through the dictionary.

#    results = geocoder.geocode(query)

#    lat = str(results[0]['geometry']['lat'])

#    lng = str(results[0]['geometry']['lng'])

#    print (f"{query} is located at:")

#    print (f"Lat: {lat}, Lon: {lng}")



#Now for JSON

weatherout = []

citydict = {}

for i in citytracker.keys():

    query = i #Here is where we grab the city name as the loop iterates through the dictionary.

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{query} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    citydict['city'] = query

    citydict['lat'] = lat

    citydict['lng'] = lng

#    citydict['temp'] = results[0]

    print(citydict)



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



#thecity = ["Miami","Atlanta","Austin"]

#for cityName in thecity:

#    data = json.load(getForecast(cityName))

#    kelvin = data['main']['feels_like']

#    human = round((kelvin - 273.15) * 9/5 + 32)

#    print("The current weather in "+cityName+f" is: {data['weather'][0]['description']}")

#    print("Right now in " + cityName + " it feels like %dºF" %human)



#Now for JSON

weatherout = []

citydict = {}

for i in citytracker.keys():

    query = i #Here is where we grab the city name as the loop iterates through the dictionary.

    results = geocoder.geocode(query)

    data = json.load(getForecast(cityName))

    kelvin = data['main']['feels_like']

    human = round((kelvin - 273.15) * 9/5 + 32)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{query} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    citydict['city'] = query

    citydict['lat'] = lat

    citydict['lng'] = lng

    citydict['feelsliketemp'] = human

    print(citydict)

    kelvin = data['main']['feels_like']

    human = round((kelvin - 273.15) * 9/5 + 32)

    weatherout.append(citydict)

#print("Right now in " + cityName + " it feels like %dºF" %human)

print("The JSON attempt looks like this:")

print(weatherout)



#Write it to a file.

out = open("cityweather.txt", "w")

#out.write(citytracker) #This doesn't work but it's close to what I would like to do. Stuck.

out.close() #Close the file when I am done. 


