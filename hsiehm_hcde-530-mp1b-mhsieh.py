# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# need to install googlemaps and opencage via "pip install -U googlemaps" and "pip install opencage" in the console
import  googlemaps
from datetime import datetime as time
from opencage.geocoder import OpenCageGeocode

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Secret keys
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Google Maps")
secret_value_1 = user_secrets.get_secret("OpenCage")
secret_value_2 = user_secrets.get_secret("OpenWeather")
        
gMaps = googlemaps.Client(key=secret_value_0)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fString = "/kaggle/input/tennis_courts.csv" # take the print out from the beginning and re-use it as a variable, rather than retyping the string each time we need it
tennisData = pd.read_csv(fString) # create a pandas dataframe and assign it to the tennisData variable
tennisData.head() # take a look at the first 5 items in the dataframe
tennisData.info() # get a quick summary of the dataframe in case the head is misleading for the actual data in the set.
# renames all of the columns to appropriately match their representation in the .csv
# we need to rename longitude at the start, or else we'll end up with 2 columns called longitude
# place it into a new dataframe so we still have the raw data if we need to go back (otherwise the code blocks here will have some errors)
col = {'longitude': 'combined', 'name' :'address', 'address' :'city', 'city' :'state', 'state' : 'zip_code', 'zip_code' : 'type', 'type' : 'count', 'count' :'clay', 'clay' : 'wall' ,'wall' : 'grass', 'grass' : 'indoor', 'indoor' : 'lights', 'lights' : 'proshop', 'proshop' : 'latitude', 'latitude' : 'longitude'}
tennis_fix = tennisData.rename(columns=col)
tennis_fix # preview the data
# remove the last column that we're not interested in currently
# put it into a new dataframe like before
tennis_strip = tennis_fix.drop(['combined'], axis=1)
tennis_strip # preview the new dataframe
# our only numeric value is really the count, and everything else is either nominal, or binary, so let's just get statistics for our count
tennis_strip['count'].describe()
tennis_strip = tennis_strip[tennis_strip['count'] != 0] # assigns the dataframe elements where the count is non-zero (aka, actually has tennis courts)
tennis_strip['count'].describe()
import matplotlib.pyplot as plt # import some plotting libraries if we need them

bins = np.arange(0, tennis_strip['count'].max() + 1.5) - 0.5 # we want to center the bins at the integers; calculations to do so
plt.figure(figsize=(20,10)) # set the x,y size of the figure so it's easier to read
data = plt.hist(tennis_strip['count'], bins) # plot a histogram using the data from our count and the bins we just created

# make some changes to the tick marks on both axes
plt.xticks(np.arange(0, tennis_strip['count'].max(), 2)) # modifying the x axis ticks so that it's easier to see where the columns align
plt.yticks(np.arange(0, data[0].max(), 500)) # do the same for y ticks
# look at some visualizations for type splits
tennis_strip['type'].value_counts().plot(kind='pie', legend=True, autopct='%.2f', fontsize=15, figsize=(10,10)) # create a pie chart using the type column and show percents; adjust font size + figure size
# create two new dataframes separating out data for each type
tennis_public = tennis_strip[tennis_strip['type'] == 'Public'] 
tennis_private = tennis_strip[tennis_strip['type'] == 'Homeowners Community']

plt.close() # close any figures from before so pyplot isn't wasting memory

fig2, ax = plt.subplots(figsize=(20,10)) # create a plot that we will reuse and set its size

pub_bins = np.arange(0, tennis_public['count'].max() + 1.5) - 0.5 # same calculations as before for our public
pri_bins = np.arange(0, tennis_public['count'].max() + 1.5) - 0.5 # " " for our private
pu_heights, pu_bins = np.histogram(tennis_public['count'], bins=pub_bins) # generate the histogram for public
pr_heights, pr_bins = np.histogram(tennis_private['count'], bins=pri_bins) # generate the histogram for private


ax.bar(pu_bins[:-1], pu_heights, facecolor='blue', alpha=0.6) # plot our public histogram
ax.bar(pr_bins[:-1], pr_heights, facecolor='red', alpha=0.5) # plot our private histogram
plt.xticks(np.arange(0, tennis_public['count'].max(), 2)) # modifying the x axis ticks so that it's easier to see where the columns align
tennis_clay = tennis_strip[tennis_strip['clay']==True] # create new dataframe for clay courts
tennis_grass = tennis_strip[tennis_strip['grass']==True] # create new dataframe for grass courts
tennis_hard = tennis_strip[tennis_strip['grass']==False] # new dataframe for hard courts, first strip out grass courts
tennis_hard = tennis_hard[tennis_hard['clay']==False] # now strip out clay courts

labels = ['clay', 'grass', 'hard'] # create labels for the bar graph
plt.barh(labels,[len(tennis_clay),len(tennis_grass),len(tennis_hard)]) # make a horizontal bar chart

clay_states = tennis_clay['state'].value_counts() # look at the state breakdown
clay_states # preview
# we can further refine to cities within each state
clay_florida = tennis_clay[tennis_clay['state'] == 'fl'] # new dataframe with only Florida data
cities_fl = clay_florida['city'].value_counts() # get counts based on city info

# do the same for New York data
clay_ny = tennis_clay[tennis_clay['state'] == 'ny']
cities_ny = clay_ny['city'].value_counts()

# print out dataframes to view information
print(cities_fl)
print(cities_ny)
import urllib.error, urllib.parse, urllib.request, json # import necessary packages

# pre-written function by Brock
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

# pre-written function by Brock, modified to use lat and long data instead
# future 
def getForecast(lati, longi):
    key = secret_value_2
    url = "https://api.openweathermap.org/data/2.5/weather?lat="+str(lati)+"&lon="+str(longi)+"&appid="+key
    print(url)
    return safeGet(url)

fl_forecasts = {}
ny_forecasts = {}

for index, values in cities_fl.items():
    coords = clay_florida[clay_florida['city'] == index][['latitude','longitude']] # create a separate dataframe for the geolocation of each clay site in Florida
    for index2, row in coords.iterrows(): # iterate through to call forecasts for each site
        data = json.load(getForecast(row['latitude'], row['longitude'])) # use geolocations to make the call
        fl_forecasts.update({index2: data['weather'][0]['description']}) # assign description into our new dictionary tiedto the site

for index2, value2 in cities_ny.items():
    coords2 = clay_ny[clay_ny['city'] == index2][['latitude','longitude']] # create a separate dataframe for the geolocation of each clay site in New York
    for index3, row2 in coords2.iterrows():
        data2 = json.load(getForecast(row2['latitude'], row2['longitude']))
        ny_forecasts.update({index3: data2['weather'][0]['description']})

clay_florida['description'] = fl_forecasts # add a new column with the weather description
clay_ny['description'] = ny_forecasts

print(len(fl_forecasts))
print(len(ny_forecasts))
public_states = tennis_public['state'].value_counts() # look at the state breakdown
public_states2 = public_states.rename('public') # preview and rename the column
# do the same for public
private_states = tennis_private['state'].value_counts()
private_states2 = private_states.rename('private') 
combined_states = pd.concat([public_states2, private_states2], axis=1)
combined_states

plt.close() 
fig3, ax3 = plt.subplots(figsize=(20,10))
ax3.scatter(combined_states['public'],combined_states['private'], c="b", marker ="o") # create a scatter plot with the public data as the x axis and private as y. modify the markers 
x = np.linspace(0, combined_states['public'].max(), 1000) # get the limits of a line based on so we can see how this compares to our pie chart values
ax3.plot(x, 0.846381*x, color='r') # plot a red line to denote our 54.16 public vs 45.84 private breakdown

i=0 # iterator
for row in combined_states.iterrows(): # iterate based on the index values for our dataframe
    ax3.annotate(row[0].upper(), (combined_states['public'][i], combined_states['private'][i]), fontsize=15) # though it's not readable, annotate the index next to each point, so we can get a better idea of wher each state falls
    i += 1 # iterate up
# function for calling Google Distance Matrix API
def getDistances(coords, sites):
    key2 = secret_value_0 # google API key
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+coords+"&destinations=" # start creating the url necessary to make the API call, adding in our origin coordinates and wrappers for destination
    for key in sites.keys(): # loop through our list of sites
            url += sites[key] + "|" # add the lat/long pair and a | char to continue adding addresses
    url = url[:-1] # remove any excess | chars that were added, since we're done adding addressed
    url += "&key="+ key2 # add our API key to make the call
    print(url)
    return safeGet(url)

mall = '3000 184th St SW, Lynnwood, WA 98037' # address of our origin
geocoder = OpenCageGeocode(secret_value_1) # create an OpenCage geocoder
results = geocoder.geocode(mall) # get the lat/long information for our origin
mall_coord = str(results[0]['geometry']['lat']) + ","+ str(results[0]['geometry']['lng']) # turn the results into a string for adding to our URL

lynnwood_tennis = tennis_public[(tennis_public['city'] == 'seattle') & (tennis_public['lights'] == True) & (tennis_public['count'] == 4) & (tennis_public['wall'] == True)] # reduce our dataframe to only courts that meet our criteria
lynnwood_coords = {} # create a dictionary to store our lat/long pairings
x = 0 # iterator
for index in lynnwood_tennis.iterrows():
    lynnwood_coords[x] = str(index[1].latitude) + "," + str(index[1].longitude) # just like our origin, turn each site into a lat/long pair
    x += 1 # increment so we're adding to the right index/key
results = json.load(getDistances(mall_coord,lynnwood_coords)) # make the API call and store our results
destinations = {} # create a dictionary for reading out our values

x = 0 # iterator
s_timeReal = "" # helper variable for tracking quickest trip
s_timeCompare = 100000 # default value so that we can quickly find the shortest travel time in seconds
saddress = "" # store quickest trip location

for address in results['destination_addresses']:
    destinations.update({address: results['rows'][0]['elements'][x]['duration']['text']}) # add the time it takes to get to that address; we could use 'text' for a readable format, but it wouldn't be comparable
    if(results['rows'][0]['elements'][x]['duration']['value'] < s_timeCompare): # check for the shortest time
        s_timeReal = results['rows'][0]['elements'][x]['duration']['text'] # assign the text to our variable so we can output something useful, rather than seconds
        s_timeCompare = results['rows'][0]['elements'][x]['duration']['value'] # reassign the quickest time thus far so we can continue comparing properly
        s_address = address # store the address
    print(f"From {mall}, it will take {destinations[address]} to get to {address}.") # as we go through, show people the addresses + time
    x += 1

print(f"The closest address that meets your criteria is {address} and it will take {s_timeReal} to get there.") # format resulting shortest distance
