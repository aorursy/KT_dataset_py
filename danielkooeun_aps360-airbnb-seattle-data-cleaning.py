import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

from re import sub

from decimal import Decimal

from sklearn.preprocessing import MultiLabelBinarizer

from scipy import stats
listings = pd.read_csv('../input/seattle/listings.csv')

ld = listings.loc[:,['id','host_is_superhost','neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'guests_included', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'cleaning_fee', 'instant_bookable', 'cancellation_policy']] 
display(listings.head())
ld.count()
ld = ld.dropna(subset=['host_is_superhost','neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'guests_included', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'instant_bookable', 'cancellation_policy']) 

ld.count()
ld['cleaning_fee'] = ld['cleaning_fee'].fillna(0)
ld['cleaning_fee'].head()
ld.head()
ld[ld['bedrooms'] > 6]
ld.loc[ld.loc[:, 'host_is_superhost'] == 't', 'host_is_superhost'] = 1

ld.loc[ld.loc[:, 'host_is_superhost'] == 'f', 'host_is_superhost'] = 0
ld.head()
# all the possible values

set(ld['neighbourhood_group_cleansed'])
neighbourhood = pd.get_dummies(ld['neighbourhood_group_cleansed'].str.lower().str.replace(' ', '_'))
neighbourhood.head()
ld_1 = pd.merge(ld, neighbourhood, left_index=True, right_index=True)

ld_1 = ld_1.drop('neighbourhood_group_cleansed', 1)
ld_1.head()
property_type = pd.get_dummies(ld_1['property_type'])
property_type.head()
ld_2 = pd.merge(ld_1, property_type, left_index=True, right_index=True)

ld_2 = ld_2.drop('property_type', 1)
ld_2.head()
room_type = pd.get_dummies(ld_2['room_type'])
room_type.head()
ld_3= pd.merge(ld_2, room_type, left_index=True, right_index=True)

ld_3 = ld_3.drop('room_type', 1)
bed_type = pd.get_dummies(ld_3['bed_type'])
bed_type.head()
ld_4= pd.merge(ld_3, bed_type, left_index=True, right_index=True)

ld_4 = ld_4.drop('bed_type', 1)
ld_4.loc[ld_4.loc[:, 'instant_bookable'] == 't', 'instant_bookable'] = 1

ld_4.loc[ld_4.loc[:, 'instant_bookable'] == 'f', 'instant_bookable'] = 0
ld_4.head()
cancellation = pd.get_dummies(ld_4['cancellation_policy'])
cancellation.head()
ld_5= pd.merge(ld_4, cancellation, left_index=True, right_index=True)

   
ld_5['guests_included'].max()
def normalizing(column):

    new_column = (column - column.min()) / (column.max() - column.min())

    return new_column
ld_5['guests_included'] = normalizing(ld_5['guests_included'])
ld_5.head()
ld_5['bathrooms'] = normalizing(ld_5['bathrooms'])
ld_5['bedrooms'] = normalizing(ld_5['bedrooms'])
ld_5['beds'] = normalizing(ld_5['beds'])
ld_5.head()
## seattle airport: 47.4502° N, 122.3088° W

airport_lat = 47.4502

airport_lon = -122.3088



## downtown: 47.6050° N, 122.3344° W

dt_lat = 47.6050

dt_lon = -122.3344



## pike place: 47.6101° N, 122.3421° W

pp_lat = 47.6101

pp_lon = -122.3421



## seattle amazon headquarter: 47.6062° N, 122.3321° W

amazon_lat = 47.6062

amazon_lon = -122.3321



## longitude and latitude in datasets

lat_data = ld_5['latitude']

lon_data = ld_5['longitude']
lat_data[1]
lon_data[1]
airport_lat
AVG_EARTH_RADIUS = 6371
def haversine_array(lat1, lng1, ld_5):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, ld_5['latitude'], ld_5['longitude']))

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arctan2(np.sqrt(d), np.sqrt(1-d))

    return h
ld_5['d_airport'] = haversine_array(airport_lat, airport_lon, ld_5)

ld_5['d_downtown'] = haversine_array(dt_lat, dt_lon, ld_5)

ld_5['d_pikeplace'] = haversine_array(pp_lat, pp_lon, ld_5)

ld_5['d_amazon'] = haversine_array(amazon_lat, amazon_lon, ld_5)

ld_5.head()
ld_5['d_airport'] = normalizing(ld_5['d_airport'])

ld_5['d_downtown'] = normalizing(ld_5['d_downtown'])

ld_5['d_pikeplace'] = normalizing(ld_5['d_pikeplace'])

ld_5['d_amazon'] = normalizing(ld_5['d_amazon'])

ld_5.head()
ld_6 = ld_5.drop('latitude', 1)

ld_6.head()
ld_7 = ld_6.drop('longitude', 1)
ld_7.head()
ld_7['price'] = ld_7['price'].replace('[\$,]','',regex=True).astype(float)

ld_7['cleaning_fee'] = ld_7['cleaning_fee'].replace('[\$,]','',regex=True).astype(float)

ld_7['price'] = normalizing(ld_7['price'])

ld_7['cleaning_fee'] = normalizing(ld_7['cleaning_fee'] )
ld_7.head()
ld_8 = ld_7.copy()

ld_8.head()
ld_8["amenities"] = ld_8["amenities"].str.lower().str.replace('{','').str.replace('}','').str.replace('"','').str.replace(' ','_').str.split(',')

ld_8.head()



mlb = MultiLabelBinarizer()

final_df = ld_8.join(pd.DataFrame(mlb.fit_transform(ld_8.pop('amenities')),

                          columns=mlb.classes_,

                          index=ld_8.index))

final_df.head()
df = final_df.loc[:,['id','guests_included', 'bathrooms', 'bedrooms', 'beds', 'price']] 
z = np.abs(stats.zscore(final_df.loc[:,['guests_included', 'bathrooms', 'bedrooms', 'beds', 'price']] ))

print(z)
df.count()
exclude_outlier = df[(z < 3).all(axis=1)]
exclude_outlier.head()
exclude_outlier1 = exclude_outlier.drop(['guests_included','bathrooms', 'bedrooms', 'beds', 'price'], 1)
exclude_outlier.count()
exclude_outlier1.head()
final = pd.merge(final_df, exclude_outlier1, how='right', on='id')
final.head()
final.count()
reviews = pd.read_csv('../input/seattle/reviews.csv')
reviews.head()
rv = reviews.loc[:, ['listing_id', 'id','comments']]
rv.head(20)
listings_and_reviews = pd.merge(rv, final, left_on = "listing_id", right_on = "id")
len(set(listings_and_reviews['listing_id']))
listings_and_reviews.head()
listings_and_reviews.iloc[:,1].count()

combined = listings_and_reviews.dropna()

combined.iloc[:,1].count()
combined.groupby('listing_id').count().head()
size = 10 # sample size 

replace = True # with replacement 

fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:] 

combined = combined.groupby('listing_id', as_index=False).apply(fn).drop(['id_x','id_y'], axis=1)
comments = combined['comments']
comments.head(10)
combined['comments'].head()
from stop_words import get_stop_words

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords

import string

from nltk.tokenize import word_tokenize 
def preprocess(sentence):

    outputSentence = sentence.lower()

    outputSentence = replaceContractions(outputSentence)

    outputSentence = removePunc(outputSentence)

    outputSentence = removeNumbers(outputSentence)

    #outputSentence = remove_non_english(outputSentence)

    return outputSentence
def replaceContractions(sentence):

    outputSentence = sentence

    outputSentence = outputSentence.replace("won't", "will not")

    outputSentence = outputSentence.replace("can\'t", "can not")

    outputSentence = outputSentence.replace("n\'t", " not")

    outputSentence = outputSentence.replace("\'re", " are")

    outputSentence = outputSentence.replace("\'s", " is")

    outputSentence = outputSentence.replace("\'d", " would")

    outputSentence = outputSentence.replace("\'ll", " will")

    outputSentence = outputSentence.replace("\'t", " not")

    outputSentence = outputSentence.replace("\'ve", " have")

    outputSentence = outputSentence.replace("\'m", " am")

    return outputSentence

def removePunc(sentence):

    removePuncTrans = str.maketrans("", "", string.punctuation)

    outputSentence = sentence.translate(removePuncTrans)

    return outputSentence
def removeNumbers(sentence):

    outputSentence = sentence

    removeDigitsTrans = str.maketrans('', '', string.digits)

    outputSentence = outputSentence.translate(removeDigitsTrans)

    return outputSentence
combined['comments'] = combined['comments'].apply(preprocess)
import re

def EngStopword(context):

    english = re.findall("[a-z]+",context)

    e_clean = [t for t in english if t not in stopwords.words('english') and len(t) is not 1]

    return e_clean
combined['comments'] = combined['comments'].apply(EngStopword)
combined['comments'].head(20)
combined.head()
combined1 = combined.groupby('listing_id')['comments'].apply(list)
combined4 = combined1.to_frame()
combined2 = combined.drop('comments', 1)
combined3 = combined2.drop_duplicates('listing_id')
combined3.head()
combined3.head()
combined4.head()
final = pd.merge(combined4, combined3 , how='inner', on='listing_id')

final.head()
final.to_csv('data1.csv')