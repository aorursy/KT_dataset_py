# import relevant libraries

import folium

from folium.plugins import MarkerCluster

import pandas as pd

import numpy as np
# load data set from kaggle, uploaded from desktop

path = ("../input/australian-coordinates-mapping/coordinates_mapping.xlsx")



all_data = pd.read_excel(path, sheet_name = 'all_data')

postcode_df = pd.read_excel(path, sheet_name = 'postcode_data') 

city_df = pd.read_excel(path, sheet_name = 'city_data')

suburb_df = all_data[all_data['suburb_type'] == 'Delivery Area']
# choose main city, purpose of selecting base map

def get_city(df):

    city_1 = False

    

    while city_1 == False:

        if city_1 == False:

            # most attempts will be included in reference table

            try: 

                input_city = input('What city are you going to look at?\n').title()



                lat = city_df.loc[city_df['city'].str.contains(input_city), 'latitude_4dec'].iloc[0]

                long = city_df.loc[city_df['city'].str.contains(input_city), 'longitude_4dec'].iloc[0]

                return lat, long

                city_1 = True

            

            # is it worth saying specifically index error? 

            except IndexError:

                print("Sorry, I'm pulling from Australia's top 97 cities and I can't find your result! \nPlease try another city.")

        

        # would this else ever occur?

        else:

            print("Sorry, I'm pulling from Australia's top 97 cities and I can't find your result! \nPlease try another city.")

            continue
# choose suburb or postcode and return coordinates, purpose of placing folium marker

def get_coordinates(df):

    loc_1 = False

    

    while loc_1 == False:

        if loc_1 == False:

            input_path = input('Do you want to select a postcode or suburb?\n').lower()

            

            if input_path == 'postcode' or input_path == 'suburb':

                if input_path == 'postcode':

                    

                    # must be integer

                    try:

                        postcode1 = int(input("What postcode are you looking for?"))

                                        

                        lat1 = postcode_df.loc[postcode_df['postcode'] == postcode1, 'latitude_4dec'].iloc[0]

                        long1 = postcode_df.loc[postcode_df['postcode'] == postcode1, 'longitude_4dec'].iloc[0]

                        return lat1, long1

                        loc_1 = True

                    

                    # value error if not integer

                    # THIS DOESN'T TAKE ME BACK TO "WHAT POSTCODE ARE YOU LOOKING FOR?"

                    except ValueError:

                        print("Sorry, postcode must be a 4-digit number!")

                        continue 

                else:

                    suburb1 = input("What suburb are you looking for?").upper()

                    

                    count_sub = suburb_df[{'suburb': suburb1}]

                    count_suburb = count_sub[count_sub.suburb == suburb1].count()['suburb']                    

                    

                    # if only 1 of suburb, give suburb coordinates

                    if count_suburb == 1: 

                        lat2 = suburb_df.loc[suburb_df['suburb'].str.contains(suburb1), 'latitude_4dec'].iloc[0]

                        long2 = suburb_df.loc[suburb_df['suburb'].str.contains(suburb1), 'longitude_4dec'].iloc[0]

                        return lat2, long2

                        loc_1 = True 

                                

                    # if more than 1 of suburb, give top 10 options

                    # TODO: I want to filter out all states not included in get_city() function, but that would mean

                    # I join two functions together, and that is bad? 

                    # ORDER SUBURB_OPTIONS BY NO. CUSTOMERS TO GIVE TOP RESULT

                    if count_suburb > 1:

                        print("\nWe've found these options for your suburb:")

                    

                        suburb_options = suburb_df[suburb_df['suburb'] == suburb1] # slice for only selected suburb

                        suburb_options.index = np.arange(1,len(suburb_options)+1) # reindex from 1

                        display(suburb_options[['suburb','postcode','state']].head(10)) # np display makes it pretty



                        option_select = int(input("Which option are you looking for? Enter the index number.\n"))



                        lat2 = suburb_options['latitude_4dec'].iloc[option_select-1] # must minus 1 due to index

                        long2 = suburb_options['longitude_4dec'].iloc[option_select-1]

                        

                        return lat2, long2

                        loc_1 = True                

            else:

                print("Please select either 'postcode' or 'suburb'")

                continue
# calculate trip distance in kilometres, using the Haversine formula. this is "as the crow flies", not by google maps via roads 

def distance(lat1, lat2, lon1,lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return int(round(12742 * np.arcsin(np.sqrt(a))))
"""

# This code would normally prompt user, but Kaggle doesn't support user input when displayed

# base map with marker cluster

map1 = folium.Map(location = get_city(postcode_df), zoom_start = 12)

marker_cluster = MarkerCluster().add_to(map1)



# plot markers

folium.Marker(location = get_coordinates(postcode_df), popup = "FILL",

             icon = folium.Icon(color = 'blue')).add_to(map1)

map1

"""
# because Kaggle doesn't support user input

print("What city are you going to look at?\nmelbourne\nDo you want to select a postcode or suburb?\nsuburb\nWhat suburb are you looking for?\nabbotsford\nWe've found these options for your suburb:\n")

suburb_options1 = suburb_df[suburb_df['suburb'] == "ABBOTSFORD"]

display(suburb_options1[['postcode','postcode','state']].head(10))

print("Which option are you looking for? Enter the index number.\n2")

print("Abbotsford is ", distance(melb[0],abbotsford[0],melb[1],abbotsford[1]),"km from Melbourne.")

melb = (-37.8145, 144.9702)

abbotsford = (-37.8017, 144.9987)



# base map with marker cluster

map2 = folium.Map(location = melb, zoom_start = 12)

marker_cluster = MarkerCluster().add_to(map2)



# plot markers

folium.Marker(location = abbotsford, popup = "FILL",

             icon = folium.Icon(color = 'blue')).add_to(map2)

map2