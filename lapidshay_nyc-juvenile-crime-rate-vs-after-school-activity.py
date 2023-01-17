import pandas as pd

import numpy as np

import math

import re

from collections import Counter

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap

from sklearn.cluster import KMeans

from wordcloud import WordCloud

%matplotlib inline
ARREST_PATH = "../input/NYPD_Arrest_Data_Year_to_Date.csv"

EDUCATIONAL_PATH = "../input/DYCD_after-school_programs.csv"
def felony_description_clean(desc):

    """Returns a clean string of input description."""

    

    # if input is np.nan

    if type(desc) is not str:

        return ''



    output = desc.lower()

    output = re.sub(r'[0-9]+|etc[.]*|unknown time|unknown|-misd.', '',output)

    output = re.sub(r'[,]+', ',',output)

    output = re.sub(r',[ ]*possession|,[ ]*possess.|,[ ]*possessi', ' possession',output)

    output = re.sub(r',[ ]*sale', ' sale',output)

    output = re.split(',|&', output)



    # replace 1st and 2nd if seperated by comma

    if len(output)>1:

        output.insert(0, output.pop(1))



    output = [word for word in output if not re.findall(r'unc',word)]

    output = ['robbery'] if 'robbery' in output else output

    output = ' '.join(output).strip().capitalize()

    output = output.replace('Ny','NY')

    output = re.sub(r'[ ]+', ' ',output)

    if output=='Obstr breath/circul':

        output = 'Obstruction of Breathing or Blood Circulation'

    if output == 'Endangering welfare child':

        output = 'Child welfare endangering'

        

    return output
def code_to_borough(code):

    """Maps a borough code to its corresponding borough name."""

    

    code_dict = {'B': 'Bronx', 'S': 'Staten Island', 'K': 'Brooklyn', 'M': 'Manhattan' , 'Q': 'Queens'}

    return code_dict[code]
# read CSV

arrest_df = pd.read_csv(ARREST_PATH, index_col=False)
# rename columns

arrest_df.rename(columns={"ARREST_BORO": "Borough", "PD_DESC": "Felony_description",  "AGE_GROUP": "Age_group"}, inplace = True)



# clean the data

arrest_df['Borough'] = arrest_df['Borough'].apply(code_to_borough)

arrest_df['Felony_description'] = arrest_df['Felony_description'].apply(felony_description_clean)



# leave only relevant columns

arrest_df= arrest_df[['Borough','Felony_description','Age_group', 'Latitude', 'Longitude']]



# drop cuplicates

arrest_df.drop_duplicates(inplace=True)
arrest_df.head()
def wordcloud(df, column):

    """Creates and shows a wordcloud created of frequencies of values in input column."""

    

    # create frequncies of felony descriptions

    felonies_freq = Counter(df[column])



    # plot wordcloud

    wordcloud = WordCloud(width = 1200, height = 800, min_font_size = 14, colormap='Dark2', prefer_horizontal=0.85)

    wordcloud.generate_from_frequencies(frequencies=felonies_freq)

    plt.figure(figsize = (20, 18), facecolor = None) 

    plt.imshow(wordcloud)
wordcloud(arrest_df, 'Felony_description')
def _address_clean(address):

    """Returns a clean address string."""

    

    # join adress to one string (if composed of a list whic was split beacues of '\n')

    address = ' '.join(address)

    

    # put whitespace between address and zip code

    address = f'{address[:-5]} {address[-5:]}'

    

    return address
def _coordianted_complete(coords, original_lat, original_lon):

    """Returns fixed Latitude and Longitude values."""

    

    # create latitude and longitude floats from coords string

    latitude_from_location, longitude_from_location = coords[1:-1].split(', ')

    latitude_from_location = float(latitude_from_location)

    longitude_from_location = float(longitude_from_location)

    

    # if latitude or longitude do not exist, populate them with coordinates extracted from 'Address_and_Coords' columns

    if np.isnan(original_lat) or np.isnan(original_lon):

        return latitude_from_location, longitude_from_location

        

    else:

        return original_lat, original_lon
def address_and_coords_fix(df):

    """Returns lists of fixed addresses, latitudes and longitudes."""



    addresses, latitudes, longitudes = [], [], []



    for tup in df.itertuples():



        # split 'Address_and_Coords' values to address (might be composed of several rows) and to coordinated tuple (string)

        *address, coords = tup.Address_and_Coords.split('\n')

        

        # clean address and add to list

        addresses.append(_address_clean(address))

        

        # complete coordiantes and add to lists

        latitude, longitude = _coordianted_complete(coords, tup.Latitude, tup.Longitude)

        latitudes.append(latitude)

        longitudes.append(longitude)

        

    return addresses, latitudes, longitudes
# read CSV

edu_df = pd.read_csv(EDUCATIONAL_PATH)
# rename columns

columns_rename_map = {

    "BOROUGH / COMMUNITY": "Borough", "PROGRAM TYPE": "Program_type",

    "PROGRAM": "Program", "SITE NAME": "Site_name", "Location 1": 'Address_and_Coords'

}

edu_df.rename(columns=columns_rename_map, inplace = True)



# drop duplicates and NaNs

edu_df.dropna(subset=['Address_and_Coords'], inplace=True)



# filter only New-York city boroughs

edu_df = edu_df[edu_df['Borough'].isin(['Bronx', 'Manhattan', 'Brooklyn', 'Queens', 'Staten Island'])]



# remove unnecessary columns

edu_df = edu_df[['Program_type', 'Program', 'Site_name', 'Borough', 'Address_and_Coords', 'Latitude', 'Longitude']]



# fix address and coordinates

addresses, latitudes, longitudes = address_and_coords_fix(edu_df)



edu_df['Address'] = addresses

edu_df['Latitude'] = latitudes

edu_df['Longitude'] = longitudes



# drop duplicates (data contain a lost of duplicates, with different names)

edu_df.drop_duplicates(subset = ['Latitude','Longitude'], inplace=True)



# drop 'Address_and_Coords' column, and change columns order

edu_df= edu_df[['Program_type', 'Program', 'Site_name', 'Borough', 'Address', 'Latitude', 'Longitude']]
edu_df.head()
# create a list of children (<18 Age_group) crime location-tuples

arrest_18_df = arrest_df[(arrest_df['Age_group']=='<18')]

arrest_18_locations = list(zip(arrest_18_df['Latitude'],arrest_18_df['Longitude']))



# create a list of after-scholl activity location-tuples

edu_positions = list(zip(edu_df['Latitude'],edu_df['Longitude']))  # create a list of locations (tuples)



# create folium map centerd on the average location of crimes

fol = folium.Map(np.mean(arrest_18_locations, axis=0), zoom_start=10, tiles = 'Stamen Terrain')  



# create Heatmap of crime locations

HeatMap(arrest_18_locations, radius = 13).add_to(fol)



# add circles to denote after-school activity locations

[folium.Circle(pos, radius=750, color='black', fill=True, fill_opacity = 0.01, weight=0.4).add_to(fol) for pos in edu_positions]



fol
def _corrected_distance(point_1, point_2, lat_corr=111.32, lon_corr=84.288):

    """

    Returns the corrected distance between 2 location (location-tuples).



    Folium uses Geographic Decimal degrees coordinate system.

    As we go farther from the equator, geographic coordinate system requires a different correction.

    The approximated corrections of measured distance in Latitude and Longitude

    in New-York city are 111.32 and 84.288 respectively.

    """



    lat_dist = (point_1[0] - point_2[0]) * lat_corr

    lon_dist = (point_1[1] - point_2[1]) * lon_corr

    return math.sqrt(lat_dist ** 2 + lon_dist ** 2)
def _in_edu_range(edu_locations, crime_locations, efficiency_range):

    """Returns a list of crime locations which are within efficiency range of at least one after-school activity location."""

    in_range_crimes = []



    for crime_location in crime_locations:

        in_range = False



        # check if crime_location is within efficiency_range of any edu-location

        for edu_location in edu_locations:

            if _corrected_distance(crime_location, edu_location) < efficiency_range:

                in_range = True

                break



        # if found, append to in_range_crimes list

        if in_range:

            in_range_crimes.append(crime_location)



    return in_range_crimes
def _borough_locations(ed_df, arr_df, borough):

    """Returns lists of all after-school activities' locations and crime locations within borough."""



    temp_edu = ed_df[ed_df['Borough'] == borough]

    edu_locs = list(zip(temp_edu['Latitude'], temp_edu['Longitude']))



    temp_arr = arr_df[arr_df['Borough'] == borough]

    arrest_locs = list(zip(temp_arr['Latitude'], temp_arr['Longitude']))



    return edu_locs, arrest_locs
def _heuristic_kmeans(crime_locations, efficiency_range, num_clusters, max_opt_groups):

    """

    Returns a list of after-school activity locations recommendations.



    Performs multiple K-Means clustering ,in a heuristic manner, as described above.

    Number of locations is determined by num_clusters.

    """

    

    new_edu_options = []



    for num in range(num_clusters, max_opt_groups):



        # create and fit k-means model with num clusters

        model = KMeans(n_clusters=num)

        model.fit(crime_locations)



        # a list of number of arrests which where out-of-range and now are in-range

        new_in_range = [len(_in_edu_range([loc], crime_locations, efficiency_range)) for loc in model.cluster_centers_]



        # append education locations, along with number of corresponding in-range crime locations

        new_edu_options.append((model.cluster_centers_, new_in_range, sum(new_in_range)))



    # extract maximum benefit group of generated locations

    max_benefit = max(new_edu_options, key=lambda x: x[2])



    # extract num_clusters-max-crimes new education locations

    new_edu = max_benefit[0][np.argsort(max_benefit[1])[-num_clusters:]]

    new_edu_scores = np.array(max_benefit[1])[np.argsort(max_benefit[1])[-num_clusters:]]



    # return the locations which yielded maximal in-range crime locations

    return new_edu, new_edu_scores
def _pop_string(dic):

    """Creates the string for folium popups."""

    

    lat = str(dic['loc'][0])[:6]

    lon = str(dic['loc'][1])[1:7]

    return f'Location:\nN{lat}/W{lon}\nScore: {dic["score"]}'
def _summary(comp_loc, edu_loc, borough, in_rng, out_rng, max_rng, rec_edu_loc, num_rec):

    """Pretty printing."""



    new_in_range = len(_in_edu_range(rec_edu_loc, out_rng, max_rng))

    output = f'{len(comp_loc)} <18 children crimes are recorded year to date in {borough}.\n'

    output += f'{len(edu_loc)} after-school programs are taking place in {borough}.\n'

    output += f'{len(in_rng)} of the crimes are committed within {max_rng} km of some program location.\n'

    output += f'{len(out_rng)} of the crimes are farther than {max_rng} km of any program location.\n'

    output += f'The new {num_rec} recommended location will bring {new_in_range} crimes into {max_rng} km range.'

    return output
def _create_map(arrest_locations, in_range_crimes, out_range_crimes, rec_edu_locations, rec_edu_scores, max_range):

    """

    Return a folium map with marked out-of-range crimes, in-range crimes and circles denoting reccomendations

    of new after-school activities' locations.

    """

    

    # create folium map centerd on the average location of crimes

    map_center_location = np.array(arrest_locations).mean(axis=0)

    ny_map = folium.Map(location=map_center_location, zoom_start=11,control_scale=True, tiles = 'Stamen Terrain')

    

    # add red markers to denote out-of-range crimes

    [folium.CircleMarker(pos, radius=1, color='red', fill=True, weight=1).add_to(ny_map) for pos in out_range_crimes]

    

    # add green markers to denote in-range crimes

    [folium.CircleMarker(pos, radius=0.8, color='green', fill=True, weight=0.8).add_to(ny_map) for pos in in_range_crimes]

    

    # create popup proporties for edu-locations

    pop_dict = {i: {'loc': loc, 'score': score} for i, (loc, score) in enumerate(zip(rec_edu_locations, rec_edu_scores))}

    

    # add circles to denote the recommendations, and their efficiency range

    [folium.Circle(

        pos, radius=max_range*1000, color='black', fill=True, weight=0.7,

        popup=_pop_string(pop_dict[i])).add_to(ny_map)

     for i, pos in enumerate(rec_edu_locations)]

    

    return ny_map
def edu_recommend(borough, max_range, num_rec, ed_df, arr_df, max_opt_groups=None):

    

    # create locations' lists of after-school activities and arrests

    edu_locations, arrest_locations = _borough_locations(edu_df, arr_df, borough)

    

    # a list of all arrest locations in range of some after-school activity

    in_range_crimes = _in_edu_range(edu_locations, arrest_locations, max_range)

    

    # a list of all arrest locations out of range of any after-school activity.

    out_range_crimes = [cr for cr in arrest_locations if cr not in in_range_crimes]

    

    max_opt_groups = max_opt_groups if max_opt_groups else num_rec + 10

    

    # find best locations to cover out of range arrests

    rec_edu_locations, rec_edu_scores = _heuristic_kmeans(out_range_crimes, max_range, num_rec, max_opt_groups)

    

    # create folium map

    ny_map = _create_map(arrest_locations, in_range_crimes, out_range_crimes, rec_edu_locations, rec_edu_scores, max_range)

    

    #Summary string generation

    summ = _summary(arrest_locations, edu_locations, borough,

                   in_range_crimes, out_range_crimes, max_range, rec_edu_locations, num_rec)

    

    return ny_map, rec_edu_locations, rec_edu_scores, summ
def location_print(rec_loc, rec_loc_scores):

    """Pretty printing."""



    print('Ranked list of recommended locations (ranked by amount of new covered crimes):')

    for rec, score in reversed(list(zip(rec_loc, rec_loc_scores))):

        lat = str(rec[0])[:6]

        lon = str(rec[1])[1:7]

        print(f'Location N {lat} / W {lon} will bring {score} crimes into an after-school activity efficency range.')
borough = 'Brooklyn'  # choose from 'Bronx', 'Manhattan', 'Brooklyn', 'Queens', or 'Staten Island'.

ny_map, rec_edu_locations, rec_edu_scores, summ = edu_recommend(borough, 0.7, 7, edu_df, arrest_18_df)
ny_map
print(summ)
location_print(rec_edu_locations, rec_edu_scores)
arrest_18_df[(arrest_18_df['Longitude'] == -73.88143295699997) & (arrest_18_df['Latitude'] == 40.67110691100004)]
borough = 'Manhattan' # choose from 'Bronx', 'Manhattan', 'Brooklyn', 'Queens', or 'Staten Island'.

ny_map, rec_edu_locations, rec_edu_scores, summ = edu_recommend(borough, 0.5, 14, edu_df, arrest_18_df)
ny_map
print(summ)
location_print(rec_edu_locations, rec_edu_scores)