from pandas.io.json import json_normalize

import folium

from geopy.geocoders import Nominatim

import requests

import pandas as pd

from bs4 import BeautifulSoup

import seaborn as sns



import requests # library to handle requests

import numpy as np # library to handle data in a vectorized manner

import random # library for random number generation



from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values



# libraries for displaying images

from IPython.display import Image 

from IPython.core.display import HTML 

    

# tranforming json file into a pandas dataframe library

from pandas.io.json import json_normalize



import time

import folium # plotting library

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', 500)



#libraries for Data preprocess

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer



#librarie for ML Clustring

from sklearn.cluster import KMeans
# Read Toronto neighbourhood profiles 2016 Data as DataFrame

df = pd.read_csv('../input/neighbourhood-profiles-2016-csv.csv')

print(df.shape)

df.head()
# create a new dataframe that contains the df swivel with the demographic characteristics in features, and the meighborhoods in index



df_final = pd.DataFrame(columns = df['Characteristic'] + ' ' + df['Topic'], index = df.columns)



for index, row in df.iterrows():

    for x in df_final.columns:

        if (row['Characteristic'] + ' ' + row['Topic']) == x:

            df_final[x] = row



# delete unnecessary fields and treat null values

df_final.drop(['_id', 'Category', 'Topic', 'Data Source'], inplace = True)

indexe_nan = df_final.isna().sum().to_frame()[df_final.isna().sum().to_frame()[0] >= 140].index

df_final.drop(indexe_nan, axis = 1, inplace = True)

df_final.columns = df_final.columns.str.strip()



# Suppression of duplications

df_final = df_final.loc[:,~df_final.columns.duplicated()]



df_final.drop(['Characteristic', 'City of Toronto'], axis = 0 , inplace = True)



# Converting character strings to float and int

df_final= df_final.apply(lambda x: x.str.replace(',',''))

df_final=df_final.apply(lambda x: x.str.replace('%',''))



for col in df_final.columns:

    if col != 'TSNS2020 Designation Neighbourhood Information' :

        df_final[col] = df_final[col].astype(float)

print(df_final.shape)

df_final.head()
# For each neighborhood I assign its coordinates.



# geolocator does not recognize all the nominations of the neighbor hood, 

# for that I had to look for the longitudes and latitudes of these neighborhoods unknown by 

# the geolocator on wikipedia and I left him the rest of the neighborhoods that he recognizes.



for index, row in df_final.iterrows():

    if index == 'Agincourt South-Malvern West' :

        df_final.loc[index, 'latitude'] =  43.7925

        df_final.loc[index, 'longitude'] =  -79.283889

    elif index == 'Bedford Park-Nortown' :

        df_final.loc[index, 'latitude'] =  43.73

        df_final.loc[index, 'longitude'] =  -79.411

    elif index == 'Cabbagetown-South St. James Town' :

        df_final.loc[index, 'latitude'] =  43.620543

        df_final.loc[index, 'longitude'] =  -79.47793

    elif index == 'Downsview-Roding-CFB' :

        df_final.loc[index, 'latitude'] =  43.732222

        df_final.loc[index, 'longitude'] =  -79.493333

    elif index == 'Mimico (includes Humber Bay Shores)' :

        df_final.loc[index, 'latitude'] =  43.612778

        df_final.loc[index, 'longitude'] =  -79.494167

    elif index == 'Beechborough-Greenbrook' :

        df_final.loc[index, 'latitude'] =  43.6943

        df_final.loc[index, 'longitude'] =  -79.4748

    elif index == 'Birchcliffe-Cliffside' :

        df_final.loc[index, 'latitude'] =  43.7089

        df_final.loc[index, 'longitude'] =  -79.2485

    elif index == 'Bridle Path-Sunnybrook-York Mills' :

        df_final.loc[index, 'latitude'] =  43.7359

        df_final.loc[index, 'longitude'] =  -79.3719

    elif index == 'Brookhaven-Amesbury' :

        df_final.loc[index, 'latitude'] =  43.6969

        df_final.loc[index, 'longitude'] =  -79.4938

    elif index == 'Clairlea-Birchmount' :

        df_final.loc[index, 'latitude'] =  43.7162

        df_final.loc[index, 'longitude'] =  -79.2828

    elif index == 'Dovercourt-Wallace Emerson-Junction' :

        df_final.loc[index, 'latitude'] =  43.663

        df_final.loc[index, 'longitude'] =  -79.441

    elif index == 'Eringate-Centennial-West Deane' :

        df_final.loc[index, 'latitude'] =  43.6599

        df_final.loc[index, 'longitude'] =  -79.5833

    elif index == 'Greenwood-Coxwell' :

        df_final.loc[index, 'latitude'] =  43.6721

        df_final.loc[index, 'longitude'] =  -79.3228

    elif index == 'Humbermede' :

        df_final.loc[index, 'latitude'] =  43.746297

        df_final.loc[index, 'longitude'] =  -79.541186

    elif index == 'Islington-City Centre West' :

        df_final.loc[index, 'latitude'] =  43.6309

        df_final.loc[index, 'longitude'] =  -79.5435

    elif index == 'Leaside-Bennington' :

        df_final.loc[index, 'latitude'] =  43.708

        df_final.loc[index, 'longitude'] =  -79.368

    elif index == 'Mount Olive-Silverstone-Jamestown' :

        df_final.loc[index, 'latitude'] =  43.739722

        df_final.loc[index, 'longitude'] =  -79.580278

    elif index == 'Parkwoods-Donalda' :

        df_final.loc[index, 'latitude'] =  43.7528

        df_final.loc[index, 'longitude'] =  -79.3264

    elif index == 'Playter Estates-Danforth' :

        df_final.loc[index, 'latitude'] =  43.68

        df_final.loc[index, 'longitude'] =  -79.349

    elif index == 'Princess-Rosethorn' :

        df_final.loc[index, 'latitude'] =  43.6700

        df_final.loc[index, 'longitude'] =  -79.5477

    elif index == 'Rockcliffe-Smythe' :

        df_final.loc[index, 'latitude'] =  43.6769

        df_final.loc[index, 'longitude'] =  -79.4894

    elif index == 'St.Andrew-Windfields' :

        df_final.loc[index, 'latitude'] =  43.7548

        df_final.loc[index, 'longitude'] =  -79.3855

    elif index == "Tam O'Shanter-Sullivan" :

        df_final.loc[index, 'latitude'] =  43.7811

        df_final.loc[index, 'longitude'] =  -79.2981

    elif index == "Thistletown-Beaumond Heights" :

        df_final.loc[index, 'latitude'] =  43.737222

        df_final.loc[index, 'longitude'] =  -79.565278

    elif index == "Westminster-Branson" :

        df_final.loc[index, 'latitude'] =  43.7856

        df_final.loc[index, 'longitude'] =  -79.4511

    elif index == "Wexford/Maryvale" :

        df_final.loc[index, 'latitude'] =  43.7613

        df_final.loc[index, 'longitude'] =  -79.3008

    elif index == "Willowridge-Martingrove-Richview" :

        df_final.loc[index, 'latitude'] =  43.6762

        df_final.loc[index, 'longitude'] =  -79.5705

    elif index == "Bay Street Corridor" :

        df_final.loc[index, 'latitude'] =  43.657291

        df_final.loc[index, 'longitude'] =  -79.384302

        

        

    elif index == "Bathurst Manor" :

        df_final.loc[index, 'latitude'] =  43.7628

        df_final.loc[index, 'longitude'] =  -79.4569

    elif index == "Bay Street Corridor" :

        df_final.loc[index, 'latitude'] =  43.7303

        df_final.loc[index, 'longitude'] =  -79.384302

    elif index == "Bedford" :

        df_final.loc[index, 'latitude'] =  43.7628

        df_final.loc[index, 'longitude'] =  -79.4114

    elif index == "Black Creek" :

        df_final.loc[index, 'latitude'] =  43.669444

        df_final.loc[index, 'longitude'] =  -79.511389

    elif index == "Briar Hill-Belgravia" :

        df_final.loc[index, 'latitude'] =  43.7037

        df_final.loc[index, 'longitude'] =  -79.4524

    elif index == "Forest Hill South" :

        df_final.loc[index, 'latitude'] =  43.6932

        df_final.loc[index, 'longitude'] =  -79.4126



    elif index == "Glenfield-Jane Heights" :

        df_final.loc[index, 'latitude'] =  43.757222

        df_final.loc[index, 'longitude'] =  -79.517778



    elif index == "Junction Area" :

        df_final.loc[index, 'latitude'] =  43.665556

        df_final.loc[index, 'longitude'] =  -79.464444

    elif index == "Kennedy Park" :

        df_final.loc[index, 'latitude'] =  43.716667

        df_final.loc[index, 'longitude'] =  -79.259722



    elif index == "Kingsway South" :

        df_final.loc[index, 'latitude'] =  43.6527

        df_final.loc[index, 'longitude'] =  -79.5072



    elif index == "Lambton Baby Point" :

        df_final.loc[index, 'latitude'] =  43.6575

        df_final.loc[index, 'longitude'] =  -79.4925



    elif index == "Malvern" :

        df_final.loc[index, 'latitude'] =  43.811667

        df_final.loc[index, 'longitude'] =  -79.231111



    elif index == "Markland Wood" :

        df_final.loc[index, 'latitude'] =  43.6336

        df_final.loc[index, 'longitude'] =  -79.5708



    elif index == "Milliken" :

        df_final.loc[index, 'latitude'] =  43.825833

        df_final.loc[index, 'longitude'] =  -79.300833



    elif index == "Morningside" :

        df_final.loc[index, 'latitude'] =  43.787

        df_final.loc[index, 'longitude'] =  -79.206



    elif index == "Roncesvalles" :

        df_final.loc[index, 'latitude'] =  43.6463

        df_final.loc[index, 'longitude'] =  -79.4491



    elif index == "Rouge" :

        df_final.loc[index, 'latitude'] =  43.820833

        df_final.loc[index, 'longitude'] =  -79.206111



    elif index == "Rustic" :

        df_final.loc[index, 'latitude'] =  43.713

        df_final.loc[index, 'longitude'] =  -79.489



    elif index == "South Parkdale" :

        df_final.loc[index, 'latitude'] =  43.640454

        df_final.loc[index, 'longitude'] =  -79.436731





    elif index == "South Riverdale" :

        df_final.loc[index, 'latitude'] =  43.66775

        df_final.loc[index, 'longitude'] =  -79.34961



    elif index == "Stonegate-Queensway" :

        df_final.loc[index, 'latitude'] =  43.630278

        df_final.loc[index, 'longitude'] =  -79.484167



    elif index == "University" :

        df_final.loc[index, 'latitude'] =  43.661667

        df_final.loc[index, 'longitude'] =  -79.395



    elif index == "Waterfront Communities-The Island" :

        df_final.loc[index, 'latitude'] =  43.620833

        df_final.loc[index, 'longitude'] =  -79.378611



    elif index == "West Humber-Clairville" :

        df_final.loc[index, 'latitude'] =  43.742

        df_final.loc[index, 'longitude'] =  -79.617



    elif index == "Weston" :

        df_final.loc[index, 'latitude'] =  43.700989

        df_final.loc[index, 'longitude'] =  -79.5197



    elif index == "Wychwood" :

        df_final.loc[index, 'latitude'] =  43.68

        df_final.loc[index, 'longitude'] =  -79.423611

        

        

    elif index == 'Blake-Jones' :

        df_final.loc[index, 'latitude'] =  43.66775

        df_final.loc[index, 'longitude'] =  -79.34961

    elif index == 'Clanton Park' :

        df_final.loc[index, 'latitude'] =  43.75

        df_final.loc[index, 'longitude'] =  -79.45

    elif index == 'Mount Pleasant East' :

        df_final.loc[index, 'latitude'] =  43.696351

        df_final.loc[index, 'longitude'] =  -79.384882

    elif index == 'Mount Pleasant West' :

        df_final.loc[index, 'latitude'] =  43.696351

        df_final.loc[index, 'longitude'] =  -79.384882

    elif index == 'North Riverdale' :

        df_final.loc[index, 'latitude'] =  43.66775

        df_final.loc[index, 'longitude'] =  -79.34961

    elif index == 'Oakwood Village' :

        df_final.loc[index, 'latitude'] =  43.6925

        df_final.loc[index, 'longitude'] =  -79.440833

    elif index == 'Danforth' :

        df_final.loc[index, 'latitude'] =  43.68

        df_final.loc[index, 'longitude'] =   -79.349

    elif index == 'Kensington-Chinatown' :

        df_final.loc[index, 'latitude'] =  43.6529

        df_final.loc[index, 'longitude'] =  -79.3980

    elif index == 'Woburn' :

        df_final.loc[index, 'latitude'] =  43.766667

        df_final.loc[index, 'longitude'] =  -79.227778

    elif index == 'Weston-Pelham Park' :

        df_final.loc[index, 'latitude'] =  43.672

        df_final.loc[index, 'longitude'] =  -79.457

    elif index == 'Mount Pleasant West' :

        df_final.loc[index, 'latitude'] =  43.672

        df_final.loc[index, 'longitude'] =  -79.457

    elif index == 'Lawrence Park South' :

        df_final.loc[index, 'latitude'] =  43.722

        df_final.loc[index, 'longitude'] =  -79.388

    elif index == 'York University Heights' :

        df_final.loc[index, 'latitude'] =  43.762

        df_final.loc[index, 'longitude'] =  -79.5

    elif index == 'Long Branch' :

        df_final.loc[index, 'latitude'] =  43.762

        df_final.loc[index, 'longitude'] =  -79.5

    elif index == 'Lawrence Park North' :

        df_final.loc[index, 'latitude'] =  43.722

        df_final.loc[index, 'longitude'] =  -79.388

    elif index == 'Lansing-Westgate' :

        df_final.loc[index, 'latitude'] =  43.757

        df_final.loc[index, 'longitude'] =  -79.417

    elif index == 'Forest Hill North' :

        df_final.loc[index, 'latitude'] =  43.7

        df_final.loc[index, 'longitude'] =  -79.416667

    elif index == 'Niagara' :

        df_final.loc[index, 'latitude'] =  43.643

        df_final.loc[index, 'longitude'] =  -79.408



    else :

        address = index

#         print(str(i) +' ' + str(index))

        geolocator = Nominatim(user_agent="foursquare_agent")

        location = geolocator.geocode(address, timeout=15)

        latitude = location.latitude

        longitude = location.longitude

        df_final.loc[index, 'latitude'] =  latitude

        df_final.loc[index, 'longitude'] =  longitude

print('Done!')      

df_final[['latitude', 'longitude']].head(10)
# Subsequently, using the foursquare API I look for all the sites that are within 1 mile 

# (1.6Km) from the neighborhoods coordinates obtained with the geolocator 

# and I generate a Data Frame that lists all these sites.



CLIENT_ID = 'AEHGFLPSQSKF4AAG4OVSJHIPSK3MKEGADCYDVLT5UTXCZBJY' 

CLIENT_SECRET = 'LF0DLT1CGXR4KTMKGWHHVZM1XWUDTHUAPA4LVIE1ASHFU1OO' 

VERSION = '20180604'

LIMIT = 100



i = 0

search_query = ''

radius = 1610



for index, row in df_final.iterrows():

    

    url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, row.latitude, row.longitude, VERSION, search_query, radius, LIMIT)

    results = requests.get(url).json()

    results



    venues = results['response']['venues']



    dataframe = json_normalize(venues)

    

    if dataframe.empty == False :



        filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']

        dataframe_filtered = dataframe.loc[:, filtered_columns]



        def get_category_type(row):

            try:

                categories_list = row['categories']

            except:

                categories_list = row['venue.categories']



            if len(categories_list) == 0:

                return None

            else:

                return categories_list[0]['name']



        dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)



        dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

        if i == 0 :

            dataframe_filtered['Neighborhoods'] = index

            df_fc = dataframe_filtered

        else:

            dataframe_filtered['Neighborhoods'] = index

            df_fc = pd.concat([df_fc, dataframe_filtered], ignore_index=True)

        i = i+1

print('Done!')

print(df_fc.shape)

df_fc.head(10)
# Now I display the neighborhood marker and their sites



venues_map = folium.Map(location=[43.6532, -79.3832], zoom_start=5) # generate map centred around the Conrad Hotel



label= ''

folium.CircleMarker(

    [latitude, longitude],

    radius=10,

    color='red',

    popup='Toronto',

    fill = True,

    fill_color = 'red',

    fill_opacity = 0.6

).add_to(venues_map)



label = ''

for lat, lng,  in zip(df_fc.lat, df_fc.lng):

    folium.CircleMarker(

        [lat, lng],

        radius=1,

        color='blue',

        popup=label ,

        fill = True,

        fill_color='blue',

        fill_opacity=0.1

    ).add_to(venues_map)

for lat, lng,  in zip(df_final.latitude, df_final.longitude):

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        color='yellow',

        popup=label ,

        fill = True,

        fill_color='yellow',

        fill_opacity=0.6,

        clustered_marker = True

    ).add_to(venues_map)



# display map

print('Done!')

venues_map

# venues_map.save('map.html')
# for analysis purposes I merge neighborhood profile data with site data for each neighborhood

# I keep that columns necessary for viewing

df_venues = df_final.merge(df_fc, left_on=df_final.index, right_on='Neighborhoods', how  = 'left')

df_venues.drop(['latitude','longitude','address','cc','city','country','Neighborhoods', 'crossStreet','distance','formattedAddress','id','labeledLatLngs','lat','lng','name','neighborhood','postalCode','state'], axis = 1, inplace = True)

print(df_venues.shape)

df_venues.head()
# I delete sites with an unknown category and keep the restaurant category

# I apply hot encoding on the result

df_venues.dropna(inplace = True)

df_restaurent = df_venues[df_venues['categories'].str.contains('Restaurant', case = False)]

df_restaurent_dum = pd.get_dummies(df_restaurent)

print(df_restaurent_dum.shape)

df_restaurent_dum.head()
%matplotlib inline

# I calculate the correlation between the features and plot the correlation between all the featires

# and a specific category of restoration.

correlation_restaurant = df_restaurent_dum.corr()
# categories_Afghan Restaurant correlation

sns.set(rc={'figure.figsize':(40,30), "axes.labelsize":100})

sns.set(font_scale = 3)  

corr = correlation_restaurant['categories_Afghan Restaurant'].to_frame().sort_values(by= 'categories_Afghan Restaurant', ascending = False).iloc[1:50,:]

sns.barplot(x=corr['categories_Afghan Restaurant'], y=corr.index)

# as we can see Ganda, gush tic and Urdu are the languages most present at the location of the Afghan restaurants,

# it is logical because these languages are spoken in the east of Africa and south east of Asia.
# categories_Halal Restaurant correlation

corr = correlation_restaurant['categories_Halal Restaurant'].to_frame().sort_values(by= 'categories_Halal Restaurant', ascending = False).iloc[1:50,:]

sns.barplot(x=corr['categories_Halal Restaurant'], y=corr.index)

# the languages that stands out for the hallal restaurant are Sindhi,

# Creol and Swampy Creed. the first is an Afghan language which explains the Muslim community, 

# the second is a street language a little distorted and the trixiemme is a language of northern canada
# categories_Jewish Restaurant correlation

corr = correlation_restaurant['categories_Jewish Restaurant'].to_frame().sort_values(by= 'categories_Jewish Restaurant', ascending = False).iloc[1:50,:]

sns.barplot(x=corr['categories_Jewish Restaurant'], y=corr.index)

# Peul and Bavarian ethnic are the most present in the neighborhoods with the most Jewish restaurant

# and the average income and also high
# count the number of categories of restaurent most present in canada

df_fc['categories'] = df_fc['categories'].fillna(value='NoN')

# df_fc.dropna(inplace = True)

df_fc['categories'].value_counts().to_frame().head(50).plot(kind= 'barh')

# The value NoN indicates no category for the site found by foursquare and that is generally present. H

# owever, Salon / Barbershop and Park are most present in Toronto
df_fc[df_fc['categories'].str.contains('Restaurant')]['categories'].value_counts().to_frame().head(50).plot(kind= 'barh')

# we can notice that the chinese restaurant, the fast food restaurant and the restaurants without specifity 

# are the categories of restaurents most spread to torornto
# To perform a clustering and find out which are the most favorable neighborhoods to open a restaurant,

# I had to choose some features that can foster income in the area.



# Population

# Population density

# Persons living alone (per cent)

# Total income: Average

# Non-permanent residents Immigran

# Youth (15-24 years)

# Working Age (25-54 years)

# females

# Males

# After-tax income

# langitude (to save the location)

# latitude (to save the location)



df_Pop = df_final[['Population, 2016 Population and dwellings','Population density per square kilometre Population and dwellings', 'Persons living alone (per cent) Family characteristics of adults','Total income: Average amount ($) Income sources', 'Non-permanent residents Immigrant status and period of immigration', 'Youth (15-24 years) Age characteristics', 'Working Age (25-54 years) Age characteristics','Total - Population aged 15 years and over by Labour force status (Females) - 25% sample data Labour force status','Total - Population aged 15 years and over by Labour force status (Males) - 25% sample data Labour force status','After-tax income: Population with an amount Income sources','latitude', 'longitude']]

df_Pop.columns = ['Population','Population density', 'Persons living alone (per cent)', 'Total income: Average', 'Non-permanent residents Immigran', 'Youth (15-24 years)', 'Working Age (25-54 years)', 'females', 'Males', 'After-tax income','langitude', 'latitude']

print(df_Pop.shape)

df_Pop.head()
# normalisation de ces features 

scaler = Normalizer()

df_Pop_norm = scaler.fit_transform(df_Pop[['Population','Population density', 'Persons living alone (per cent)', 'Total income: Average', 'Non-permanent residents Immigran', 'Youth (15-24 years)', 'Working Age (25-54 years)', 'females', 'Males', 'After-tax income']])

pd.DataFrame(df_Pop_norm).head()
# calculates the score of each neighborhood

score = np.sum(df_Pop_norm, axis=1)

df_Pop = df_Pop.copy()

df_Pop.loc[:,'score'] = score

df_Pop.head()
# keep the 50 best neighborhoods according to the scor, then join the result with the sites available in these neighborhoods 

# by keeping that restaurants.

df_Pop = df_Pop.sort_values(by = 'score', ascending = False).head(50)

df_venues = df_Pop.merge(df_fc[df_fc['categories'].str.contains('Restaurant|NoN')], left_on=df_Pop.index, right_on='Neighborhoods', how  = 'left')

print(df_venues.shape)

df_venues.head()
# count the number of pat neighborhood restaurants and add it as a new feature for the training dataset.

# I removed a 1 for the values NoN, so that they are not counted as a restaurant

df_Pop.loc[:,'Number_of_Restaurant'] = df_venues.groupby(['Neighborhoods']).count()['Population'].to_frame()['Population']

df_Pop['Number_of_Restaurant'] = df_Pop['Number_of_Restaurant']-1

(df_Pop.shape)

df_Pop.head()
# Fit the K Means on the learning data and assign a label to each neighborhood

X = df_Pop[['langitude', 'latitude', 'score','Number_of_Restaurant']]

scaler = StandardScaler()

X= scaler.fit_transform(X)

KM = KMeans(n_clusters=4, random_state=0).fit(X)

df_Pop.loc[:,'labels'] = KM.labels_
# assign a color to each cluster and display it on the map



colors = ['red', 'blue', 'green', 'yellow']



this_map = folium.Map(prefer_canvas=True)



def plotDot(point):

    '''input: series that contains a numeric named latitude and a numeric named longitude

    this function creates a CircleMarker and adds it to your this_map'''

    folium.CircleMarker(location=[point.langitude, point.latitude],

                        radius=1,

                        color = np.array(colors)[int(point.labels)],

                        label = point.index,

                        weight=12).add_to(this_map)



#use df.apply(,axis=1) to "iterate" through every row in your dataframe

df_Pop.apply(plotDot, axis = 1)





#Set the zoom to the maximum possible

this_map.fit_bounds(this_map.get_bounds())



#Save the map to an HTML file

# this_map.save('html_map_output/simple_dot_plot.html')



this_map
print(df_Pop[df_Pop.labels == 0 ].shape)

print('average Number_of_Restaurant cluster 1 : '+ str(df_Pop[df_Pop.labels == 0 ]['Number_of_Restaurant'].mean()))

print('average score cluster 1 : ' + str(df_Pop[df_Pop.labels == 0 ]['score'].mean()))

df_Pop[df_Pop.labels == 0 ].head()
print(df_Pop[df_Pop.labels == 1 ].shape)

print('average Number_of_Restaurant cluster 2 : '+ str(df_Pop[df_Pop.labels == 1 ]['Number_of_Restaurant'].mean()))

print('average score cluster 2 : '+ str(df_Pop[df_Pop.labels == 1 ]['score'].mean()))

df_Pop[df_Pop.labels == 1 ].head()
print(df_Pop[df_Pop.labels == 2 ].shape)

print('average Number_of_Restaurant cluster 3 : '+ str(df_Pop[df_Pop.labels == 2 ]['Number_of_Restaurant'].mean()))

print('average score cluster 3 : '+ str(df_Pop[df_Pop.labels == 2 ]['score'].mean()))

df_Pop[df_Pop.labels == 2 ].head()
print(df_Pop[df_Pop.labels == 3 ].shape)

print('average Number_of_Restaurant cluster 4 : ' +str(df_Pop[df_Pop.labels == 3 ]['Number_of_Restaurant'].mean()))

print('average score cluster 4 : '+ str(df_Pop[df_Pop.labels == 3 ]['score'].mean()))

df_Pop[df_Pop.labels == 3 ].head()
# I scrap WikiPedia to associate a borough to each neighborhood



url="https://en.wikipedia.org/wiki/List_of_city-designated_neighbourhoods_in_Toronto"



page = requests.get(url)



soup = BeautifulSoup(page.content, 'lxml')



tbl = soup.find('table',{'class':'wikitable sortable'})

df_borough = pd.read_html(str(tbl))[0]



print(df_borough.shape)

df_borough.head(5)
df_result = df_Pop.merge(df_borough[['City-designated area', 'Former city/borough']], left_on=df_Pop.index, right_on='City-designated area', how  = 'left')

df_result.index = df_Pop.index

print(df_result.shape)

df_result.head()