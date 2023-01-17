# basic imports

import pandas as pd

import numpy as np



# plotting libraries and magics

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



# to calculate distance between two coordinates

from geopy.distance import geodesic



# garbage collector

import gc



# preprocessing

from sklearn.preprocessing import LabelEncoder



# modeling requirements

from sklearn.model_selection import train_test_split # to split the data into train and validation sets

from sklearn.metrics import r2_score # eval metric for this competetion



from sklearn.ensemble import RandomForestRegressor



# enabling multiple outputs for Jupyter cells

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'



# for fancy looping

from tqdm import tqdm





# --------- Setting some configs and variables --------- #

# to display all the columns instead of ... in the output (when there are many columns)

pd.set_option('display.max_columns', None)



# Declaring the PATH for all input data

PATH = "../input/night-life-challenge-trivago/"



# our random seed for the model

SEED = 42 # because, why not! ¯\_(ツ)_/¯
# to show a tabular report of the missing data in a given dataset

def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

#     return(np.transpose(tt))

    return tt



# get a list of categorical and numeric variables for a dataset

def get_num_cat_cols(df):

    """

    Returns two lists, one for categorical variables and one for numeric variables

    """

    cat_vars = [col for col in df.columns if df[col].dtype in (['object', 'O'])]

    num_vars = [col for col in df.columns if df[col].dtype not in (['object', 'O'])]

    

    return cat_vars, num_vars



# to find out if two lists have any common members

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if (a_set & b_set): 

        return True 

    else: 

        return False
# to display wordclouds



from wordcloud import WordCloud, STOPWORDS



# adding the locations as stop words too - since they don't add value to our wordcloud

stopwords = ["Greece", "Hong", "Kong", "China", "Thessaloniki", "Los", "Angeles", "Amsterdam", "'"] + list(STOPWORDS) 



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=SEED # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
# load the data into data frames

hotel = pd.read_csv(PATH + 'hotels.csv')

poi = pd.read_csv(PATH + 'pois.csv')



# let's take a peek

hotel.head(3)

poi.head(3)
# function to display missing values and other details about a dataframe - we will use this later to look at the poi dataframe.

def get_dataframe_info(df):

    # checking how many numerical and categorical columns we have

    cat_cols, num_cols = get_num_cat_cols(df)

    print(f'There are {len(num_cols)} numeric columns and {len(cat_cols)} categorical columns')

    

    print('\n******** Missing data report ********')

    print(missing_data(df))

    

    # let's see the distribution of the data as well

    print('\n',df.describe())
# hotel dataframe details

get_dataframe_info(hotel)
# gc.collect()
# function to display column-wise reports for a dataframe

def column_data_distribution(df):

    

    # let's plot the distribution for the numeric columns

    cat_cols, num_cols = get_num_cat_cols(df)

    for col in num_cols:

        print(f'\n****** Stats for {col} column ******')

        print(f'\nThere are {df[col].nunique()} unique values.')

        sns.distplot(df[col].dropna().sort_values()); 

        plt.show();

        

    # doing the same for the categorical columns

    for col in cat_cols:

        print(f'\n****** Stats for {col} column ******')

        print(f'\nThere are {df[col].nunique()} unique values.')

        sns.countplot(df[col].dropna().sort_values()); 

        plt.show();
# let's replace all NANs with 0 - under the assumption that if a particular value is not known, chances are very less that that feature is there.

hotel = hotel.fillna(0)
column_data_distribution(hotel)
hotel.drop('club_club_hotel', axis=1, inplace=True)
# how many poi_types do we have?

poi_types = set(poi['poi_types'].str.cat(sep=" ,").replace(' ,', ',').split(','))

# poi_types



# also, let's save the  poi_types as a list in a new column in poi df

poi['poi_types_list'] = [list(p.split(', ')) for p in poi['poi_types']]



# from the above, let's build two sets of poi_types, one for nightlife friendly types, and the other for the anti-nightlife types.

nightlife_poi_types = ['Pub', 'Bar', 'Disco', 'Nightclub', 'Food & Drink', 'Event', 'Entertainment', 'Bowling', 'Café', 'Casino', 'Game Centers']

anti_nightlife_poi_types = ['Religious', 'Nature', 'Architectural Buildings', 'Historic Sites', 'Lake', 'Museums', 'Spas', 'Classes / Workshops'

                           'Art Galleries', 'Botanical Gardens', 'Golf Area', 'National Parks', 'Parks', 'Palaces / Castles', 'Zoos / Aquariums']



# let's add two flags to the poi dataframe: nightlife_ok, nightlife_not_ok

poi['nightlife_ok'] = [int(common_member(p, nightlife_poi_types)) for p in poi['poi_types_list']] # 1 when the poi_type is present in nightlife_poi_types, else 0

poi['nightlife_not_ok'] = [int(common_member(p, anti_nightlife_poi_types)) for p in poi['poi_types_list']] # 1 when the poi_type is present in anti_nightlife_poi_types, else 0
# how many unique city_ids do we have in poi and hotel dataframes?

print(f'There are {hotel["city_id"].nunique()} unique cities in hotel df, and {poi["city_id"].nunique()} unique cities in poi df')

# 4 in both - so, we are good to do an inner join to merge the two dataframes



# let's now merge the hotel dataframe with the poi dataframe - using the city_id.

hotel_poi = hotel.merge(poi, on='city_id', how='inner')

print(f'Shape of merged df: {hotel_poi.shape}')



# also, let's rename the latitude and longitude columns for clarity (we shouldn't waste brain power on remembering if '_x' was for hotel or poi!)

hotel_poi.rename(columns={'longitude_x':'longitude_hotel', 'latitude_x':'latitude_hotel', 'longitude_y':'longitude_poi', 'latitude_y':'latitude_poi'}, inplace=True)



# let's see how the merged data looks like

hotel_poi.head(3)
# wordcloud for the points of interest that are nightlife_ok

show_wordcloud(hotel_poi[hotel_poi['nightlife_ok']==1].poi_types)
# wordcloud for the points of interest that are nightlife__not_ok

show_wordcloud(hotel_poi[hotel_poi['nightlife_not_ok']==1].poi_types)
# # verifying if the above steps worked

# hotel_poi[hotel_poi.nightlife_ok == 1].head(3)

# hotel_poi[hotel_poi.nightlife_not_ok == 1].head(3)
# let's pull the latitudes and longitudes into 4 series

hotel_latitudes = hotel_poi['latitude_hotel']

hotel_longitudes = hotel_poi['longitude_hotel']

poi_latitudes = hotel_poi['latitude_poi']

poi_longitudes = hotel_poi['longitude_poi']



# now, we can calculate the distance between the poi and the hotel

distance = []

for i in tqdm(range(hotel_poi.shape[0])):

    distance.append(geodesic((hotel_latitudes[i], hotel_longitudes[i]), (poi_latitudes[i], poi_longitudes[i])).meters)

print('Done!')



# adding the distances to the main dataframe

hotel_poi['distance'] = distance
# # pickle the above processed file - to save time in future runs

# PICKLE_NAME = '../input/hotel_poi.pkl'

# hotel_poi.to_pickle(PICKLE_NAME)



# # loading the pickle file

# hotel_poi = pd.read_pickle(PICKLE_NAME)

# print('Pickle loaded successfully!')
hotel_poi.distance.describe()

hotel_poi.distance_to_center.describe()
# Creating a flag to show the number of nightlife_ok pois within 500m

nightlife_ok_poi_500m = hotel_poi[(hotel_poi['nightlife_ok']==1) & (hotel_poi['distance']<=500)].groupby('hotel_id').nightlife_ok.count().reset_index()





# we'll do the same for nightlife_not_ok pois

nightlife_not_ok_poi_500m = hotel_poi[(hotel_poi['nightlife_not_ok']==1) & (hotel_poi['distance']<=500)].groupby('hotel_id').nightlife_not_ok.count().reset_index()



# merging the above two dataframes into one

poi_500m = nightlife_ok_poi_500m.merge(nightlife_not_ok_poi_500m, on='hotel_id', how='inner')
# adding the count data to the hotel dataframe

hotel = hotel.merge(poi_500m, on='hotel_id', how='left')

hotel.head()
# let's check if there are any NANs in the new columns

hotel.nightlife_ok.isnull().any()

hotel.nightlife_not_ok.isnull().any()
# replace the NANs with 0.

hotel.fillna(0, inplace=True)
# to be continued

# implement this for the geo-data: https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d

# to get the shapefiles, just google them! - or check out the files in the dataset we created :)
hotel.columns
# we want to consider the 'hotel_type'. Let's label encode that.

le = LabelEncoder()

hotel['hotel_type_encoded'] = le.fit_transform(hotel['hotel_type'])
# taking the columns we want for the model in a separate dataframe

features = hotel.loc[:,['hotel_type_encoded', 'car_park', 'attraction_hotel', 'beach_front_hotel',

       'convention_hotel', 'spa_hotel', 'country_hotel', 'airport_hotel',

       'senior_hotel', 'eco_friendly_hotel', 'party_people', 'business_people',

       'honeymooners', 'singles', 'large_groups', 'family_hotel',

       'gay_friendly', 'wifi_lobby', 'wifi_room', 'nightlife_ok',

       'nightlife_not_ok']]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=99) # 99, because for now, we don't know how many clusters we should take
wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(features)

    wcss.append(kmeans.inertia_)





plt.figure(figsize=(12, 5));

plt.title("WCSS / K Chart", fontsize=18);

plt.plot(range(1,15),wcss,"-o");

plt.grid(True);

plt.xlabel("Amount of Clusters",fontsize=14);

plt.ylabel("Inertia",fontsize=14);

plt.xticks(range(1,20));

plt.tight_layout();

plt.show();
# clustering the hotels

kmeans = KMeans(n_clusters=2, random_state=SEED);

features["labels"] = kmeans.fit_predict(features);
gc.collect()
# let's add the predicted classes to the original dataframe for validation

hotel['labels'] = features['labels']
plt.figure(figsize=(24,4));



plt.suptitle("K Means Clustering: K=2",fontsize=20);



plt.subplot(1,5,1);

plt.title("Hong Kong",fontsize=16);

plt.scatter(hotel.longitude[(hotel.labels == 0) & (hotel.city_id==31497)],hotel.latitude[(hotel.labels == 0) & (hotel.city_id==31497)], label='class 0')

plt.scatter(hotel.longitude[(hotel.labels == 1) & (hotel.city_id==31497)],hotel.latitude[(hotel.labels == 1) & (hotel.city_id==31497)], label='class 1')

plt.legend();



plt.subplot(1,5,2);

plt.title("Thessaloniki",fontsize=16);

plt.scatter(hotel.longitude[(hotel.labels == 0) & (hotel.city_id==14121)],hotel.latitude[(hotel.labels == 0) & (hotel.city_id==14121)], label='class 0')

plt.scatter(hotel.longitude[(hotel.labels == 1) & (hotel.city_id==14121)],hotel.latitude[(hotel.labels == 1) & (hotel.city_id==14121)], label='class 1')

plt.legend();



plt.subplot(1,5,3);

plt.title("Los Angeles",fontsize=16);

plt.scatter(hotel.longitude[(hotel.labels == 0) & (hotel.city_id==14257)],hotel.latitude[(hotel.labels == 0) & (hotel.city_id==14257)], label='class 0')

plt.scatter(hotel.longitude[(hotel.labels == 1) & (hotel.city_id==14257)],hotel.latitude[(hotel.labels == 1) & (hotel.city_id==14257)], label='class 1')

plt.legend();



plt.subplot(1,5,4);

plt.title("Amsterdam",fontsize=16);

plt.scatter(hotel.longitude[(hotel.labels == 0) & (hotel.city_id==27561)],hotel.latitude[(hotel.labels == 0) & (hotel.city_id==27561)], label='class 0')

plt.scatter(hotel.longitude[(hotel.labels == 1) & (hotel.city_id==27561)],hotel.latitude[(hotel.labels == 1) & (hotel.city_id==27561)], label='class 1')

plt.legend();



plt.subplots_adjust(top=0.8);

plt.show();
# # for K=3

# plt.figure(figsize=(24,4));

# plt.suptitle("K Means Clustering: K=3",fontsize=20);



# plt.subplot(1,5,1);

# plt.title("Hong Kong",fontsize=16);

# plt.scatter(hotel.longitude[(hotel.labels_3 == 0) & (hotel.city_id==31497)],hotel.latitude[(hotel.labels_3 == 0) & (hotel.city_id==31497)], label='class 0')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 1) & (hotel.city_id==31497)],hotel.latitude[(hotel.labels_3 == 1) & (hotel.city_id==31497)], label='class 1')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 2) & (hotel.city_id==31497)],hotel.latitude[(hotel.labels_3 == 2) & (hotel.city_id==31497)], label='class 2')

# plt.legend();



# plt.subplot(1,5,2);

# plt.title("Thessaloniki",fontsize=16);

# plt.scatter(hotel.longitude[(hotel.labels_3 == 0) & (hotel.city_id==14121)],hotel.latitude[(hotel.labels_3 == 0) & (hotel.city_id==14121)], label='class 0')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 1) & (hotel.city_id==14121)],hotel.latitude[(hotel.labels_3 == 1) & (hotel.city_id==14121)], label='class 1')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 2) & (hotel.city_id==14121)],hotel.latitude[(hotel.labels_3 == 2) & (hotel.city_id==14121)], label='class 2')

# plt.legend();



# plt.subplot(1,5,3);

# plt.title("Los Angeles",fontsize=16);

# plt.scatter(hotel.longitude[(hotel.labels_3 == 0) & (hotel.city_id==14257)],hotel.latitude[(hotel.labels_3 == 0) & (hotel.city_id==14257)], label='class 0')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 1) & (hotel.city_id==14257)],hotel.latitude[(hotel.labels_3 == 1) & (hotel.city_id==14257)], label='class 1')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 2) & (hotel.city_id==14257)],hotel.latitude[(hotel.labels_3 == 2) & (hotel.city_id==14257)], label='class 2')

# plt.legend();



# plt.subplot(1,5,4);

# plt.title("Amsterdam",fontsize=16);

# plt.scatter(hotel.longitude[(hotel.labels_3 == 0) & (hotel.city_id==27561)],hotel.latitude[(hotel.labels_3 == 0) & (hotel.city_id==27561)], label='class 0')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 1) & (hotel.city_id==27561)],hotel.latitude[(hotel.labels_3 == 1) & (hotel.city_id==27561)], label='class 1')

# plt.scatter(hotel.longitude[(hotel.labels_3 == 2) & (hotel.city_id==27561)],hotel.latitude[(hotel.labels_3 == 2) & (hotel.city_id==27561)], label='class 2')

# plt.legend();



# plt.subplots_adjust(top=0.8);

# plt.show();
hotel[hotel.labels == 1]
# let's pull the hotel_ids for class 0 and 1

class_0_hotel = hotel[hotel['labels']==0].hotel_id.unique()

class_1_hotel = hotel[hotel['labels']==1].hotel_id.unique()
# hotel[hotel.hotel_id.isin(class_0_hotel)]

# hotel[hotel.hotel_id.isin(class_1_hotel)]
# let's see what the class 0 and 1 are about

plt.title('Wordcloud for label 0')



# we want the poi types for pois that are less than 500m away

show_wordcloud(hotel_poi[(hotel_poi.hotel_id.isin(class_0_hotel)) & (hotel_poi.distance <= 500)].poi_types)
plt.title('Wordcloud for label 1')



# we want the poi types for pois that are less than 500m away

show_wordcloud(hotel_poi[(hotel_poi.hotel_id.isin(class_1_hotel)) & (hotel_poi.distance <= 500)].poi_types)
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(random_state=SEED, verbose=0)
bgm.fit(features);

hotel['labels_bgm'] = bgm.predict_proba(features);
import geopandas as gpd

MAP_PATH_HONGKONG = "../input/night-life-challenge-trivago/hong_kong_poi/hong_kong_poi.shp"
# the shapefiles are not giving us what we want at the moment. We'll see if we can find shapefiles that give us results to work with. 

# We'll work on that in a later version.



# # set the filepath and load in a shapefile

# # map_amsterdam = gpd.read_file(PATH + 'Amsterdam_shapefile.csv')

# map_hk = gpd.read_file(MAP_PATH_HONGKONG)



# # check data type so we can see that this is not a normal dataframe, but a GEOdataframe

# map_hk.head()

# # now let's preview what our map looks like with no data in it

# map_hk.plot();