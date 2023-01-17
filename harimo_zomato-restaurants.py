# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import ast

import operator

from matplotlib import cm

from itertools import cycle, islice

%matplotlib inline

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

import matplotlib.cm

from numpy import meshgrid

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
zom = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
# Head of the dataFrame

zom.head()
# Describe

zom.describe()
# Info

zom.info()
# Columns

zom.columns
# Removing url, phone column

zom.drop(['url', 'phone', 'address'], axis = 1, inplace = True)
# Location

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['location'], order=zom['location'].value_counts().index,hue=zom['book_table'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
# Location

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['location'], order=zom['location'].value_counts().index,hue=zom['online_order'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()

zom['online_order'].value_counts()
# Lets get the boundary of Bangalore first

boundary = pd.read_csv('../input/banglore-locations/bangalore_loc.csv', usecols = ['Lat', 'Long'])
# Function: plot_map

# Description: Takes a dataframe - location, lat, long, markersize and make a plot.

#              Boundary of Bangalore is drawn using boundary data frame thats listed above  

def plot_map(dfr,markersize = 5):    

    plt.figure(figsize=(20,20))

    map = Basemap(projection='aeqd', lon_0 = 77.5, lat_0 = 12.8, width = 150000, height = 170000, resolution='l') # set res=h

    map.drawmapboundary(fill_color='cyan')

    map.etopo()

    map.drawcoastlines()

    map.drawcountries()

    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

    scale = 0.00002

    for i in range(0,len(boundary)):

        x, y = map(boundary.ix[i,'Long'], boundary.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Red', markersize=3)

    for i in range(0,len(dfr)):

        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Green', markersize=markersize, alpha = 0.6)

    plt.show()
# Location column

zom['location'].unique()
# Getting the rough Latitude and Longitudes of the locations from Google

place = {'location': [

                   'Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',

                   'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',

                   'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',

                   'Nagarbhavi', 'Bannerghatta Road', 'BTM', 'Kanakapura Road',

                   'Bommanahalli', 'CV Raman Nagar', 'Electronic City', 'HSR',

                   'Marathahalli', 'Sarjapur Road', 'Wilson Garden', 'Shanti Nagar',

                   'Koramangala 5th Block', 'Koramangala 8th Block', 'Richmond Road',

                   'Koramangala 7th Block', 'Jalahalli', 'Koramangala 4th Block',

                   'Bellandur', 'Whitefield', 'East Bangalore', 'Old Airport Road',

                   'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'RT Nagar',

                   'MG Road', 'Brigade Road', 'Lavelle Road', 'Church Street',

                   'Ulsoor', 'Residency Road', 'Shivajinagar', 'Infantry Road',

                   'St. Marks Road', 'Cunningham Road', 'Race Course Road',

                   'Commercial Street', 'Vasanth Nagar', 'HBR Layout', 'Domlur',

                   'Ejipura', 'Jeevan Bhima Nagar', 'Old Madras Road', 'Malleshwaram',

                   'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block',

                   'Majestic', 'Langford Town', 'Central Bangalore', 'Sanjay Nagar',

                   'Brookefield', 'ITPL Main Road, Whitefield',

                   'Varthur Main Road, Whitefield', 'KR Puram',

                   'Koramangala 2nd Block', 'Koramangala 3rd Block', 'Koramangala',

                   'Hosur Road', 'Rajajinagar', 'Banaswadi', 'North Bangalore',

                   'Nagawara', 'Hennur', 'Kalyan Nagar', 'New BEL Road', 'Jakkur',

                   'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal',

                   'Kengeri', 'Sankey Road', 'Sadashiv Nagar', 'Basaveshwara Nagar',

                   'Yeshwantpur', 'West Bangalore', 'Magadi Road', 'Yelahanka',

                   'Sahakara Nagar', 'Peenya'

                ],

        'Lat':  [

                    12.9255, 12.9406, 12.9537, 12.9308,

                    12.9044, 12.9149, 12.9756,

                    12.9070, 12.9063, 12.9716, 12.9647,

                    12.9719, 12.8052, 12.9166, 12.5462,

                    12.9030, 12.9793, 12.8440, 12.9121,

                    12.9569, 12.8549, 12.9482, 12.9578, 

                    12.9352, 12.9415, 12.9661,

                    12.9363, 13.0528, 12.9315,

                    12.9304, 12.9698, 13.0012, 13.1986,

                    12.9784, 12.9265, 12.9970, 13.0196,

                    12.9766, 12.5824, 12.9712, 12.9751,

                    12.9817, 12.5820, 12.9857, 12.9832,

                    12.9723, 12.9892, 12.9615, 

                    12.9822, 12.9920, 13.0191, 12.9610, 

                    12.9385, 12.9642, 12.9851, 13.0055, 

                    12.9889, 13.0159, 12.9382, 

                    12.9767, 12.9570, 12.9716, 13.0369,

                    12.9655, 12.9698,

                    12.9698, 13.0170,

                    12.9247, 12.9286, 12.9352, 

                    12.9359, 12.9982, 13.0104, 12.9375,

                    13.0422, 13.0359, 13.0240, 13.0292, 13.0631,

                    13.0163, 12.9718, 12.9836, 13.0354,

                    12.8997, 12.9941, 13.0068, 12.9880,

                    13.0250, 12.9747, 12.9750, 13.1186,

                    13.0623, 13.0285

                ],

        'Long': [

                    77.5468, 77.5738, 77.5434, 77.5802,

                    77.5649, 12.9149, 77.5354,

                    77.5521, 77.5857, 77.5946, 77.5768,

                    77.5127, 77.5788, 77.6101, 77.4199,

                    77.6242, 77.6642, 77.6739, 77.6446,

                    77.7011, 77.7881, 77.5972, 77.5993,

                    77.6200, 77.6178, 77.5949,

                    77.6128, 77.5419, 77.6300,

                    77.6784, 77.7500, 77.6183, 77.7066,

                    77.6408, 77.6362, 77.6144, 77.5968,

                    77.5993, 77.3157, 77.5978, 77.3450,

                    77.6284, 77.3450, 77.6057, 77.6047,

                    77.6012, 77.5932, 77.6157,

                    77.6083, 77.5943, 77.6465, 77.6387,

                    77.6308, 77.6581, 77.6434, 77.5692,

                    77.5740, 77.6379, 77.6228, 

                    77.5713, 77.6028, 77.5946, 77.5785,

                    77.7185, 77.7500,

                    77.7500, 77.7044,

                    77.6207, 77.6291, 77.6244, 

                    77.6088, 77.5530, 77.6482, 77.4472,

                    77.6136, 77.6431, 77.6433, 77.5709, 77.6207,

                    77.6785, 77.6552, 77.6797, 77.5988,

                    77.4827, 77.5860, 77.5813, 77.5375,

                    77.5340, 77.5701, 77.2231, 77.5975,

                    77.5871, 77.5197

                ]

       }



loc = pd.DataFrame(place, columns = ['location', 'Lat', 'Long'])
zom = pd.merge(zom, loc, how='left', on='location')

zom.head()
# Lets plot the locations first

plot_map(loc,markersize = 5)
# listed_in(city) is just a subset of locations column, we can use either one depending on the usage during further analysis

zom['listed_in(city)'].unique()
data = {'location': [

                   'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',

                   'Brigade Road', 'Brookefield', 'BTM', 'Church Street',

                   'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',

                   'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',

                   'Koramangala 4th Block', 'Koramangala 5th Block',

                   'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',

                   'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',

                   'Old Airport Road', 'Rajajinagar', 'Residency Road',

                   'Sarjapur Road', 'Whitefield'

                ],

        'Lat':  [

                    12.9255, 12.8070, 12.9406, 12.9304,

                    12.5824, 12.9698, 12.9166, 12.9751,

                    12.8440, 12.9970, 12.9121, 12.9784,

                    12.9308, 12.9063, 13.0240, 13.0159,

                    12.9315, 12.9352, 

                    12.9382, 12.9363, 12.9711,

                    13.0055, 12.9569, 12.9766, 13.0292,

                    12.9600, 12.9982, 12.9661,

                    12.8600, 12.9698

                ],

        'Long': [  

                    77.5468, 77.5787, 77.5738, 77.6784,

                    77.3157, 77.7500, 77.6101, 77.6047,

                    77.6739, 77.6144, 77.6446, 77.6408,

                    77.5802, 77.5857, 77.6433, 77.6379,

                    77.6300, 77.6200, 

                    77.6228, 77.6128, 77.5978,

                    77.5692, 77.7011, 77.5993, 77.5709,

                    77.6460, 77.5530, 77.5949,

                    77.7860, 77.7500

                ]

       }

listedInCity = pd.DataFrame(data, columns = ['location', 'Lat', 'Long'])
# Lets plot the locations first

plot_map(listedInCity, markersize = 5)
# Now that we have an rought idea of the locations, we can plot further features

# Lets look at Ratings columns first

# Changing Rating column's format

zom['rate'] = zom['rate'].str.split('/', n = 1, expand = True)[0]

zom['rate'] = zom['rate'].str.split(' ', n = 1, expand = True)[0]

zom['rate'].unique()
# Lets look at 'NEW' rating

zom[zom['rate']=='NEW']['votes'].unique()
plt.figure(figsize=(15,7), dpi =100)

ax = zom['rate'].value_counts().sort_index().plot.bar()

ax.set_xlabel('Rating')

ax.set_ylabel('Count')
zom.info()
# Since ratings is granular eg, 2.0, 2.1 ..., lets approximate them to the nearst round number

# -, New, 1.8 - 2.9 : Poor (0)

# 3.0 - 3.9 : Average (1)

# 4.0 - 4.4 : Good (2)

# 4.5 - 4.9 : Great (3)



d = {

     '-': 0, 'NEW' : 0, '1.8': 0, 

     '2.0' : 0, '2.1' : 0, '2.2' : 0, '2.3' : 0, '2.4' : 0, '2.5' : 0, '2.6' : 0, '2.7' : 0, '2.8' : 0, '2.9' : 0, 

     '3.0' : 1, '3.1' : 1, '3.2' : 1, '3.3' : 1, '3.4' : 1, '3.5' : 1, '3.6' : 1, '3.7' : 1, '3.8' : 1, '3.9' : 1,   

     '4.0' : 2, '4.1' : 2, '4.2' : 2, '4.3' : 2, '4.4' : 2, 

     '4.5' : 3, '4.6' : 3, '4.7' : 3, '4.8' : 3, '4.9' : 3   

    }

zom['NewRating'] = zom['rate'].map(d)
plt.figure(figsize=(15,7), dpi =100)

ax = zom['NewRating'].value_counts().sort_index().plot.bar()

ax.set_xlabel('Rating')

ax.set_ylabel('Count')

ax.set_xticklabels(('Poor','Average', 'Good', 'Great'))
# Lets add ratings column to our listedInCity dataFrame

listedInCity = pd.DataFrame(data, columns = ['location', 'Lat', 'Long'])

for r in list(zom['NewRating'].unique()):

    for c in list(zom['listed_in(city)'].unique()):

        listedInCity.loc[listedInCity['location'] == c, r] = len(zom[(zom['listed_in(city)'] == c) & (zom['NewRating'] == r)])

listedInCity.drop(np.nan, axis = 1, inplace = True)

listedInCity.head()
# Lets plot different ratings on our plot

for r in [0.0, 1.0, 2.0, 3.0]:

    if r in [0.0, 2.0, 3.0]:

        scale = 0.05

    elif r == 1.0:

        scale = 0.02

    plt.figure(figsize=(20,20))

    map = Basemap(projection='aeqd', lon_0 = 77.5, lat_0 = 12.8, width = 150000, height = 170000, resolution='l') # set res=h

    map.drawmapboundary(fill_color='cyan')

    map.etopo()

    map.drawcoastlines()

    map.drawcountries()

    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

    

    plt.title("{} Star Rating".format(r))

    for i in range(0,len(boundary)):

        x, y = map(boundary.ix[i,'Long'], boundary.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Red', markersize=3)

    for i in range(0,len(listedInCity)):

        x, y = map(listedInCity.ix[i,'Long'], listedInCity.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Green', markersize=int(listedInCity.ix[i,r]*scale), alpha = 0.6)

    plt.show()
zom.columns
# Ratings vs book_table

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['NewRating'], order=zom['NewRating'].value_counts().index.sort_values(),hue=zom['book_table'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
# Ratings vs book_table

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['rate'], order=zom['rate'].value_counts().index.sort_values(),hue=zom['book_table'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
# Ratings vs online_order

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['NewRating'], order=zom['NewRating'].value_counts().index.sort_values(),hue=zom['online_order'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
# Ratings vs online_order

plt.figure(figsize=(15,7), dpi =100)

plot = sns.countplot(zom['rate'], order=zom['rate'].value_counts().index.sort_values(),hue=zom['online_order'])

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
# book_table, online_order

fig,axes = plt.subplots(1,2,figsize=(15,5))

sns.countplot(zom['book_table'], ax = axes[0])

sns.countplot(zom['online_order'], ax = axes[1])

#axes[1] = sns.countplot(zom['NewRating'], order=zom['NewRating'].value_counts().index.sort_values(),hue=zom['online_order'])

#plt.tight_layout()
zom.columns
# Votes column

zom[zom['votes'] == zom['votes'].max()]
# Restaurant with the most VOTES

zom.iloc[zom['votes'].idxmax()]
# Top 5 restaurants based by vote counts

top5votes = zom.iloc[zom['votes'].sort_values(ascending = False).index]['name'].head(30).unique()

top5votes
# Their total num of votes

for re in top5votes:

    print(re, ' : ', zom[zom['name'] == re]['votes'].sum())
# List of restaurants with highest number of votes in each listed_in(city)

lc = list(zom['listed_in(city)'].unique())

print('City', '             ', 'Name of Restaurant', '            ', 'Number of Votes')

for c in lc:

    temp = zom.loc[zom[zom['listed_in(city)'] == c]['votes'].idxmax()]

    print(c, "        ", temp['name'], "        ", temp['votes'])
# List of restaurants with highest number of votes in each listed_in(city)

lc = list(zom['listed_in(city)'].unique())

print('City', '             ', 'Name of Restaurant', '            ', 'Number of Votes')

for c in lc:

    temp = zom.loc[zom[zom['listed_in(city)'] == c]['votes'].idxmin()]

    print(c, "        ", temp['name'], "        ", temp['votes'])    
zom.columns
# Plotting book_table vs votes

fig, axes = plt.subplots(3, 1, figsize=(15,17), dpi =100)

zom[zom['book_table'] == 'Yes']['votes'].plot.hist(bins = 500, ax = axes[0])

zom[zom['book_table'] == 'No']['votes'].plot.hist(bins = 300, ax = axes[1])

zom.loc[(zom['book_table'] == 'No') & (zom['votes'] > 0)]['votes'].plot.hist(bins = 500, ax = axes[2])
# Plotting online_order vs votes

fig, axes = plt.subplots(2, 1, figsize=(15,17), dpi =100)

zom[zom['online_order'] == 'Yes']['votes'].plot.hist(bins = 500, ax = axes[0])

zom[zom['online_order'] == 'No']['votes'].plot.hist(bins = 300, ax = axes[1])
zom.columns
zom.head()
zom['rest_type'].unique()
def rest_count(zom_df):

    rest_dict = {'Unknown': 0}

    for j in range(len(zom_df)):

        if zom_df['rest_type'][j] is not np.nan:

            rests = zom_df['rest_type'][j].split(', ')

            for rest in rests:  #iterates over each cuisine style in the list

                if rest in rest_dict:

                    rest_dict[rest] += 1

                else :

                    rest_dict[rest] = 1

        else:

            rest_dict['Unknown'] +=1

    return (rest_dict)
restaurants = rest_count(zom)

restaurants = pd.Series(restaurants) 

print('-----------------------------------------')

print('      Type of Restaurants split-up')

print('-----------------------------------------')

print(restaurants.sort_values(ascending = False))
# Plot the various rest types

plt.figure(figsize=(15,7), dpi =100)

restaurants.sort_values(ascending = False).plot.bar()

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
def plot_city_restaurants(city_rest):

    fig, ax = plt.subplots(1,1, figsize = (7,7))

    ax = city_rest.sort_values(ascending = False).head().plot.pie(shadow = True)

    ax.set_title(city)
print('---------------------TOP 5 restaurant types per location-------------------------------------------')

for city in list(zom['listed_in(city)'].unique()):

    print(city)

    city_rest = rest_count(zom[zom['listed_in(city)'] == city].reset_index())

    city_rest = pd.Series(city_rest)

    print(city_rest.sort_values(ascending = False).head())

    print('----------------------------------------------------------------')

    plot_city_restaurants(city_rest)

    
# Cuisines

def cuisine_count(zom_df):

    cuisine_dict = {'Unknown': 0}

    for j in range(len(zom_df)):

        if zom_df['cuisines'][j] is not np.nan:

            styles = zom_df['cuisines'][j].split(', ')

            for style in styles:  #iterates over each cuisine style in the list

                if style in cuisine_dict:

                    cuisine_dict[style] += 1

                else :

                    cuisine_dict[style] = 1

        else:

            cuisine_dict['Unknown'] +=1

    print("Total number of different cuisine styles ('unknown' included) :", len(cuisine_dict))

    return(cuisine_dict)
cuisines = cuisine_count(zom)

cuisines = pd.Series(cuisines) 

print('-----------------------------------------')

print('      Type of Cuisines split-up')

print('-----------------------------------------')

print(cuisines.sort_values(ascending = False))
# Plot the various cuisines

plt.figure(figsize=(15,7), dpi =100)

cuisines.sort_values(ascending = False).plot.bar()

plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

plt.tight_layout()
def plot_city_cuisine(city_cui):

    fig, ax = plt.subplots(1,1, figsize = (7,7))

    ax = city_cui.sort_values(ascending = False).head(10).plot.pie(shadow = True)

    ax.set_title(city)
print('---------------------TOP 5 Cuisines types per location-------------------------------------------')

for city in list(zom['listed_in(city)'].unique()):

    print(city)

    city_cui = cuisine_count(zom[zom['listed_in(city)'] == city].reset_index())

    city_cui = pd.Series(city_cui)

    print(city_cui.sort_values(ascending = False).head())

    print('----------------------------------------------------------------')

    plot_city_cuisine(city_cui)