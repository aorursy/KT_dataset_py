citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
citytracker['Seattle']=citytracker['Seattle']+17500

print(str(citytracker['Seattle'])+" residents in Seattle") #use 'run all' to run code to avoid continual addition
citytracker['Los Angeles']=45000

print(citytracker['Los Angeles'])
print('Denver: '+ str(citytracker['Denver']))
for key in citytracker:

    print(key)
for x in citytracker:

    print((x)+': '+str(citytracker[x]))
if 'New York'in citytracker:

    print(('New York')+': '+str(citytracker['New York']))

else:

    print('Sorry, that is not in the City Tracker.')



if 'Atlanta'in citytracker:

    print(('Atlanta')+': '+str(citytracker['Atlanta']))

else:

    print('Sorry, that is not in the City Tracker.')

potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
for x in potentialcities:

    if x in citytracker:

        print((x)+': '+str(citytracker[x]))

    else:

        print('0')
for x in citytracker:

    print(x+','+str(citytracker[x]))
import os



### Add your code here



f=open("popreport.csv","w")

f.write("city"+","+"pop"+"\n")

for x,y in citytracker.items():

    f.write(x+','+str(y)+"\n")



f.close()





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey") # make sure this matches the Label of your key

key1 = secret_value_0





from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)



citycoordinates={}



# let's find coordinates for all the city in citytracker

for x in citytracker:

    query = x  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (x+": Lat: %s, Lon: %s" % (lat, lng))

    

    # Within for loop, create new dictionary called 'citycoordinates'with list nested

    # New dictionary will be in this format: citycoordinates={cityname:[lat,lng]}

    citycoordinates[x]=[lat,lng]

# let's print out new dictionary to check

print('citycoordinates = '+ str(citycoordinates))







# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darkSkyKey") # make sure this matches the Label of your key



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

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



# Create parent dictionary named cityforecast that houses the forecast data from darksky. 

# Structure of parent dictionary as follows:

# cityforecast={'city':{'citycoordinates':[lat,lng],'datetime':b,'condition':c,'temperature':d,'forecast':e}}



cityforecast={}



# Use for loop to get forecast for each city in citycoordinates dictionary according to their lat and lng 

for x,y in citycoordinates.items():

    data = json.load(getForecast(y[0],y[1]))

    current_time = datetime.datetime.now() 

    # Put city forecast data as values into parent dictionary - cityforecast, nested inside each city

    cityforecast[x]={}

    cityforecast[x]['Coordinates']=y

    cityforecast[x]['Datetime']="Retrieved at: %s" %current_time

    cityforecast[x]['Condition']=data['currently']['summary']

    cityforecast[x]['Temperature']=data['currently']['temperature']

    cityforecast[x]['Forecast']=data['minutely']['summary'] 

    

# Let's print to double-check

print('cityforecast = '+ str(cityforecast))



# Save dictionary to JSON file , similar to step 10 in assignment

import json



with open('cityforecast.json','w')as outfile:

    json.dump(cityforecast, outfile)
