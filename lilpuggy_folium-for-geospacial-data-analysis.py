# import libraries

import numpy as np 

import pandas as pd 

import folium as fl

import matplotlib.pyplot as plt

import seaborn as sns
#create DataFrame

df = pd.read_csv("../input/bournemouth-venues/bournemouth_venues.csv")
df.head(10)
# rename columns

df = df.rename(columns = {'Venue Latitude':'latitude','Venue Longitude': 'longitude', 'Venue Category': 'category','Venue Name':'place'})

df.columns
venues = df.category # variable so I don't have to type as much later





#  Lists of Categories for mapping the new column "Venue General Categories"

restaurant_extras = ['Sandwich Place', 'Diner', 'Pizza Place', 'Noodle House', 'Burger Joint', 'Indian Restaurant', 'English Restaurant', 'Fast Food Restaurant', 'French Restaurant']

cafe_extras = ['Coffee Shop', 'Ice Cream Shop', 'Café', 'Bubble Tea Shop', 'Dessert Shop']

bar_extras = ['Pub', 'Nightclub', 'Brewery', ]

indoor_recreation_extras = ['Multiplex', 'Theater', 'Arts & Entertainment']

outdoor_recreation_extras = ['Park', 'Plaza', 'Beach', 'Garden', 'Other Great Outdoors', 'Scenic Lookout']

educational_extras = ['Art Museum', 'Aquarium']

retail_extras = ['Clothing Store', 'Grocery Store']

transit_extras = ['Train Station', 'Bus Stop', 'Platform']





#  Creating individual dataframes for each group to be used in FacetGrid()

restaurant_data = df[(venues.isin(restaurant_extras)) & (venues.str.contains('Restaurant'))]

cafe_data = df[venues.isin(cafe_extras)]

bar_data = df[(venues.str.contains('Bar')) | (venues.isin(bar_extras))]

indoor_recreation_data = df[venues.isin(indoor_recreation_extras)]

outdoor_recreation_data = df[venues.isin(outdoor_recreation_extras)]

educational_data = df[venues.isin(educational_extras)]

retail_data = df[venues.isin(retail_extras)]

gym_data = df[venues.str.contains('Gym')]

transit_data = df[venues.isin(transit_extras)]

hotel_data = df[venues.str.contains('Hotel')]
#  Prepping variables for looping

dataframe_list = [restaurant_data, cafe_data, bar_data, indoor_recreation_data, outdoor_recreation_data,

         educational_data, retail_data, gym_data, transit_data, hotel_data]

dataframe_names = ['Restaurant', 'Cafe', 'Bar', 'Indoor Recreation', 'Outdoor Recreation', 'Educational',

                  'Retail', 'Gym', 'Transit', 'Hotel']



#  Check that the lists are of equal length

if len(dataframe_list) - len(dataframe_names) != 0:

    print('ERROR: Number of dataframes is not equal to number of dataframe names')
#  Rename rows with the General Category Names using a for loop

df['general_category'] = df['category']

for i in range(len(dataframe_list)):

    df['general_category'] = df.apply(lambda row: dataframe_names[i] \

                                        if row['general_category'] in (list(dataframe_list[i]['category'])) \

                                        else row['general_category'], \

                                        axis=1)

#print(df['general_category'].head(10))



#  Checking to make sure I didn't miss any categories

#frames = [restaurant_data, cafe_data, bar_data, indoor_recreation_data, outdoor_recreation_data,

#         educational_data, retail_data, gym_data, transit_data, hotel_data]

#all_data = pd.concat(frames)



#list1 = all_data['category'].unique()

#list2 = df['category'].unique()

#print(list(set(list2) - set(list1)))

# Different categories

print(df.category.value_counts().iloc[0:11])

print('\n')

print("total categories :",df.category.value_counts().shape)

fig = plt.figure(figsize = (20,5))

sns.countplot(df["category"][0:10])
# Folium map

map = fl.Map([50.720913,-1.879085],zoom_start = 15)
def add_to_map(data, color):

    #  Separating the hotel locations and converting each attribute into list

    lat = list(data["latitude"])

    lon = list(data["longitude"])

    place = list(data["place"])

    cat = list(data["category"])

    for lt,ln,pl,cat in zip(lat,lon,place,cat):

        fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = color)).add_to(map)
#  list of colors to use in folio

colors = ['darkred', 'darkblue', 'darkgreen', 'cadetblue',

         'darkpurple', 'lightgray', 'pink', 'lightblue', 'lightgreen',

         'gray', 'black']



#  looping the function and adding each group to the map

for i in range(len(dataframe_list)):

    add_to_map(dataframe_list[i], colors[i])

    

map
