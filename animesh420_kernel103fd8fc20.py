# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

from IPython.display import display, HTML

from geopy.geocoders import Nominatim

from geopy.distance import distance

from geopy.distance import geodesic

import os

import matplotlib.cm as cm

import matplotlib.colors as colors

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading zomato dataframe

from IPython.display import display

zomato = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

display(zomato.head())

location_coordinates = dict()
# Reading locations of each zomato circle ie neighborhood

zomato = zomato[zomato.location.map(lambda x: x is not np.nan)]

zomato['location'] = ['Koramangala' if 'koramangala' in x.lower() else x for x in zomato['location']]

import time





def outlier_handling(s):

    if s == "CV Raman Nagar":

        return "C V Raman Nagar"

    elif s == "Rammurthy Nagar":

        return "Ramamurthy Nagar"

    elif s == "Sadashiv Nagar":

        return "Sadashiva Nagar"

    elif s == "Bellandur":

        return "Bellandur Central"

    elif s == "Electronic City":

        return "Electronic City Phase 1"

    else:

        return s





location_coordinates = dict()

geolocator = Nominatim(user_agent="ny_explorer")



def get_location():

    exceptionRaised = False

    unique_locations = sorted(zomato['location'].unique())

    """

    To avoid following locations:

    1. North Bangalore 2. East Bangalore 3. South Bangalore 4. West Bangalore

    """ 

    unique_locations = [x for x in unique_locations if 'bangalore' not in x.lower()]

    i = 0

    zdf = pd.DataFrame()

    if 'zdf.csv' in os.listdir():

        zdf = pd.read_csv('./zdf.csv', index_col=0)



    while i < len(unique_locations):

        try:

            s = outlier_handling(unique_locations[i])

            print(s)

            if zdf is not None and s in zdf.index:

                print('Found {} loaded'.format(s))

                i += 1

                continue



            address = '{}, Bangalore'.format(s)



            location = geolocator.geocode(address)

            location_coordinates[s] = (location.latitude, location.longitude)

            zdf.loc[unique_locations[i], 'lats'] = location.latitude

            zdf.loc[unique_locations[i], 'longs'] = location.longitude

            print('Index = {}, Address = {} is located at lat = {}, lon = {}'.format(i, address, location.latitude,

                                                                                     location.longitude))

            i += 1

        except Exception as e:

            print(e)

            if type(e) == AttributeError:

                print(s, e)

                return

            time.sleep(5)



    return zdf

try:

    zdf = pd.read_csv("/kaggle/input/zdfdata/zdf.csv", index_col=0)

    

except Exception as e:

    print(e)

    zdf = get_location() 

display(zdf.head())



def get_distance(df, loc1, loc2):

    coords1 = df.loc[loc1, 'lats'], df.loc[loc1, 'longs']

    coords2 = df.loc[loc2, 'lats'], df.loc[loc2, 'longs']

    return geodesic(coords1, coords2).km



def getcolumns(n):

    f = ['1st', '2nd', '3rd']  

    return f if n < 4 else f  + ['{}th'.format(i) for i in range(4, n + 1)] 

def find_nearest(n=5):                                                                                                                                        

    columns = getcolumns(n)   

    unique_locations = zdf.index.unique()                                                             

    nearest = zdf.copy().dropna()                                                                              

    for location in unique_locations:                                                                 

        distanceSeries = pd.Series([get_distance(nearest, location, loc) for loc in unique_locations],    

                                   index=unique_locations)                                            

        nearest_five = distanceSeries.nsmallest(n=n + 1).index[1:]                                    

                                                                                                      

        for col, val in zip(columns, nearest_five):                                                   

            nearest.loc[location, col] = val                                                          

                                                                                                      

    return nearest.dropna()                                                                                    

nearest = find_nearest()

display(nearest[nearest.index.map(lambda x: x[0] == 'J')])
zom_loc = zomato.copy()

zom_loc = zom_loc[zom_loc.location.isin(zdf.index)]

dflist = []

zdf.dropna(axis=0)

for locs in zom_loc['location'].unique():

    df_to_change = zom_loc[zom_loc.location == locs]

    df_to_change.insert(0,'Lat',[zdf.loc[locs, 'lats']] * df_to_change.shape[0])

    df_to_change.insert(0,'Long',[zdf.loc[locs, 'longs']] * df_to_change.shape[0])

    dflist.append(df_to_change)

zom_loc = pd.concat(dflist)

print(zom_loc.location.unique()[:5])

print(zom_loc.rest_type.unique()[:5])
zloc_vs_rt = zom_loc[zom_loc.rest_type.map(lambda x: type(x) == str)]

zloc_vs_rt = zloc_vs_rt[zloc_vs_rt.rate.map(lambda x: type(x) == str and '/' in x)]

zloc_vs_rt.rate = zloc_vs_rt.rate.apply(lambda x: float(x.split('/')[0]))

allRestTypes = sorted(zloc_vs_rt['rest_type'].unique())

allLocations = sorted(zloc_vs_rt['location'].unique())

recozom_rating = pd.DataFrame(columns=allRestTypes, index=allLocations).fillna(0)

recozom_frequency = pd.DataFrame(columns=allRestTypes, index=allLocations).fillna(0)

for i, locality in enumerate(allLocations):

    df1 = zloc_vs_rt[(zloc_vs_rt.location == locality)]



    for index, row in df1.iterrows():

        try:

            rating = row['rate']

            rest_type = row['rest_type']

            recozom_frequency.loc[locality, rest_type] += 1

            recozom_rating.loc[locality, rest_type] += rating / 5

        except Exception as e:

            print(e)





for index in recozom_rating.index:



    for col in recozom_rating.columns:

        if recozom_frequency.loc[index, col] != 0:

            recozom_rating.loc[index, col] /= recozom_frequency.loc[index, col]

            recozom_rating.loc[index, col] *= 5

recozom_frequency.dropna(inplace=True)

recozom_rating.dropna(inplace=True)

display(recozom_frequency.head())

display(recozom_rating.head())
col = 'listed_in(type)'

allRestTypes = sorted(zloc_vs_rt[col].unique())

allLocations = sorted(zloc_vs_rt['location'].unique())

recozom_listin_frequency = pd.DataFrame(columns=allRestTypes, index=allLocations).fillna(0)

recozom_listin_votes = pd.DataFrame(columns=allRestTypes, index=allLocations).fillna(0)

for restType in allRestTypes:



    df = zloc_vs_rt[zloc_vs_rt[col] == restType]

    resttypecols = df['rest_type'].unique()

    unique_locations = df['location'].unique()

    for locality in unique_locations:

        reco_f = recozom_frequency.loc[locality, resttypecols]

        recozom_listin_frequency.loc[locality, restType] = sum(reco_f.values)

        

 

recozom_listin_frequency.dropna(inplace=True)

display(recozom_listin_frequency.head())

def check_name(location):

    if location not in recozom_listin_frequency.index:

        possibles = [x for x in recozom_listin_frequency.index if x.startswith(location[0])]

        possibles = ','.join(possibles)

        print(possibles)

        msg= """Please choose a correct location. 

        Locations starting from letter {} are: {}""".format(location[0],possibles)

        raise Exception(msg) 

    else:

        return True

def find_type_of_restaurant(location,num_nearest=5,bestOptionNum=2,displayOption='table'):

    getString = lambda x: ','.join([y for y in x if type(y) == str])

    col = 'listed_in(type)'

    # check if valid location is passed in

    assert check_name(location) is True

    

    # check nearby locations as specified

    n = num_nearest

    nearest = find_nearest(n)

    columns = getcolumns(n)

    nearby_places = list(nearest.loc[location, columns].values)

    print(nearby_places, location)

    locationsToConsider = [location] + nearby_places

    print('Searching for restaurant to open at {} and nearby places at {}'.format(location, getString(nearby_places)))

    

    # Show best options ( popular in neighborhood and not available at nearby location)

    allRestaurants = recozom_listin_frequency.loc[locationsToConsider]

    collist = list(allRestaurants.columns)

    print('Best to open:')

    for i, cols in enumerate(collist):

        series = allRestaurants.loc[:, cols].nsmallest(1)

        print('{}. {} at {}'.format(i,cols, series.index[0]))

    

    # Show details for restaurant sub types

    # Show details for option(Choose a number) 

    bestOption = bestOptionNum

    assert bestOption < len(collist), 'Input a number less than {}'.format(len(collist))

    bestOption = collist[bestOption]

    print('Showing sub restaurant types for chosen option {}'.format(bestOption))



    subtypes = sorted(zloc_vs_rt[zloc_vs_rt[col] == bestOption]['rest_type'].unique())

    if len(subtypes) == 0:

        print('Sorry no sub type found.')

    else:

        print('Subtypes found successfully.')

    subRestaurants_freq = recozom_frequency.loc[locationsToConsider, subtypes].dropna()

    subRestaurants_rating = recozom_rating.loc[locationsToConsider, subtypes].dropna()

    bestSubTypes = subRestaurants_rating.idxmax(axis=1)

    geolocator = Nominatim(user_agent="ny_explorer")

    # Display option :: Enter 'log' to see a log or 'table' to see a table.

    option = displayOption

    

    if option == 'log':

 

        # Display information

        locationsToOpen = subRestaurants_freq.loc[:, bestSubTypes.values].idxmin()

        print('\n### Recommendations ### ::')

        for subrestType, location in locationsToOpen.items():

            print('.'*25)

            print('*** Recommend to open {}  at {} ***'.format(subrestType, location))

            best_location = bestSubTypes.index[np.where(bestSubTypes == subrestType)][0]

            dfn = zloc_vs_rt[(zloc_vs_rt.rest_type==subrestType) & (zloc_vs_rt.location==best_location)]

            dfn = dfn[dfn.rate == dfn.rate.max()]

            cols_to_disp = ['rate','name','cuisines','dish_liked','approx_cost(for two people)','address']

            dfn = dfn.loc[:, cols_to_disp].sample()





            for index, row in dfn.iterrows():



                print('Example of Higest rated restaurant {} with metrics (located at {}):'.format(subrestType,best_location))

                print(' 1.Restaurant Name : {}'.format(row[cols_to_disp[1]]))

                print(' 2.Rate : {}'.format(row[cols_to_disp[0]]))

                print(' 3.Cusinies : {}'.format(row[cols_to_disp[2]]))

                print(' 4.Dish Liked : {}'.format(row[cols_to_disp[3]]))

                print(' 5.Cost for two : {}'.format(row[cols_to_disp[4]]))

                print(' 6.Address : {}'.format(row[cols_to_disp[-1]]))

    else:

        columns = ['Recommend to open(Subrest type)', 'location to open', 'example restaurant', 'location of example',

                   'rating','cuisines','dish_liked','cost for two','Address of Example']

        df_to_display = pd.DataFrame(columns=columns)

         

        # Display information

        locationsToOpen = subRestaurants_freq.loc[:, bestSubTypes.values].idxmin()

        print('\n### Recommendations ### ::')

        for i,(subrestType, location) in enumerate(locationsToOpen.items()):

#             print('.'*25)

#             print('*** Recommend to open {}  at {} ***'.format(subrestType, location))

            df_to_display.loc[i, columns[0]] = subrestType

            df_to_display.loc[i, columns[1]] = location

            

            best_location = bestSubTypes.index[np.where(bestSubTypes == subrestType)][0]

            dfn = zloc_vs_rt[(zloc_vs_rt.rest_type==subrestType) & (zloc_vs_rt.location==best_location)]

            dfn = dfn[dfn.rate == dfn.rate.max()]

            cols_to_disp = ['rate','name','cuisines','dish_liked','approx_cost(for two people)','address']

            dfn = dfn.loc[:, cols_to_disp].sample()





            for index, row in dfn.iterrows():



                df_to_display.loc[i,columns[2]] = row[cols_to_disp[1]]

                df_to_display.loc[i,columns[3]] = best_location

                df_to_display.loc[i,columns[4]] = row[cols_to_disp[0]]

                df_to_display.loc[i,columns[5]] = row[cols_to_disp[2]]

                df_to_display.loc[i,columns[6]] = row[cols_to_disp[3]]

                df_to_display.loc[i,columns[7]] = row[cols_to_disp[4]]

                df_to_display.loc[i,columns[8]] = row[cols_to_disp[-1]]



                

        df_to_display.insert(0,'Restaurant type',[bestOption]*df_to_display.shape[0])

        display(df_to_display)

                

        

find_type_of_restaurant('BTM')

    
col = 'listed_in(type)'

# display(list(zloc_vs_rt.columns))

zloc = zloc_vs_rt.copy().loc[:, ['location', 'votes',col]]

zloc = zloc.groupby(['location', col])['votes'].sum().reset_index().pivot(index=col, columns='location', values='votes')

zvotes = zloc.fillna(0)

display(zvotes)
def check_restaurant_type(restaurant_type):

    if restaurant_type not in recozom_listin_frequency.columns:

        possibles = list(recozom_listin_frequency.columns)

        possibles = ','.join(possibles)

        msg= """Please choose a correct Restaurant type. 

        Available locations are: {}""".format(possibles)

        raise Exception(msg) 

    else:

        return True

def find_name_of_location(restaurant_type,min_req=5,nearest_neighborhood=6,min_popular=7):

    assert check_restaurant_type(restaurant_type) is True

    #Minimum number of recommendations required.

    n = min_req

    locationSeries = recozom_listin_frequency.loc[:, restaurant_type].sort_values(ascending=True)

    # Enter number of nearest neighborhoods to check for.

    k = nearest_neighborhood

    # Enter minimum number of popular restaurant locations to show.

    m = min_popular

    nearest = find_nearest(k)

    location_df = nearest.copy().loc[locationSeries.index].drop(['lats', 'longs'], 1)

    location_df['Total'] = np.zeros(location_df.shape[0])

    cols = location_df.columns

    for index, row in location_df.iterrows():

        total = 0

        for col in cols:

            try:

                val = recozom_listin_frequency.loc[row[col], restaurant_type]

            except Exception as e:

                val = 0

            location_df.loc[index,col] = val

    location_df['Total'] = location_df.sum(axis=1)

    location_df = location_df.sort_values(by=['Total'])

    location_df.insert(0,'self',locationSeries.values)

    

    print('Displaying data (top {}) where {} is least available in neighborhoods'.format(n,restaurant_type))

    display(location_df.head(n))

    

    print('Displaying most popular (top {}) destinations for restaurant type {} (by votes)'.format(m,restaurant_type))

    display(zvotes.loc[restaurant_type,].sort_values(ascending=False).to_frame().head(m))

    





find_name_of_location('Buffet') 
