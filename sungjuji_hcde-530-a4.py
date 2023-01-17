citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
#call value attached to key 'Chicago'

citytracker['Chicago']
#replace value by assigning new value to the key 'Seattle'

citytracker['Seattle'] = 17500

citytracker['Seattle']
#Add new key called 'los angeles' and assign value of 45000

citytracker['Los Angeles'] = 45000

#call citytracker dictionary to check if value is added

citytracker
#create a parameter called 'population' and have it call population value for Denver from the dictionary

population = citytracker['Denver']

#print out outcome using string formatting

print("Denver: %d" %population)



print("\n")



#another method

print('Denver:', citytracker['Denver'])
#print keys in citytracker

for city in citytracker:

    print (city)
#assign 'city' and 'populations' as variables using citytracker.items() and print the values

for city, populations in citytracker.items():

    print(city, ":", populations)



print("\n")



#another method found through Google search (not formatted)

for item in citytracker.items():

    print (item)

    

print("\n")

#another method

for item in citytracker:

    print (item, ': ', citytracker[item])
#If New York key is found in citytracker, print "New York: " and get its value

#else, print "sorry....."



if 'New York' in citytracker:

    print("New York: ", citytracker['New York'])

else:

    print('Sorry, that is not in the City Tracker')

    

######for Atlanta#######



if 'Atlanta' in citytracker:

    print("Atlanta: ", citytracker['Atlanta'])

else:

    print('Sorry, that is not in the City Tracker')
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee', 'Denver']



#check if potential city list is found in citytracker

#if value is true, print key and its value

#if value is false, print '0'

for x in potentialcities:

    if x in citytracker:

        print(x, citytracker[x])

    else:

        print("0")

#i'm not sure if I understood the first part clearly

#print(citytracker.lstrip()) -----> ???



print(citytracker)

for item in citytracker:

    print (item, ': ', citytracker[item])
##code source:https://stackoverflow.com/questions/10373247/how-do-i-write-a-python-dictionary-to-a-csv-file



import csv



### Add your code here



with open('popreport.csv', 'w') as f:

    w = csv.writer(f)

    w.writerows(citytracker.items())



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("sjkey") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Denver'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))



with open('citydata.json', 'w') as outfile:

    json.dump(data, outfile)
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("darksky") # make sure this matches the Label of your key



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



data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])


