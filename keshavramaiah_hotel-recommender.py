import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from keras.callbacks import EarlyStopping

import math

import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

hotel_details=pd.read_csv('../input/hotel-recommendation/Hotel_details.csv',delimiter=',')

hotel_rooms=pd.read_csv('../input/hotel-recommendation/Hotel_Room_attributes.csv',delimiter=',')

hotel_cost=pd.read_csv('../input/hotel-recommendation/hotels_RoomPrice.csv',delimiter=',')
hotel_details.head()
hotel_rooms.head()
del hotel_details['id']

del hotel_rooms['id']

del hotel_details['zipcode']
hotel_details=hotel_details.dropna()

hotel_rooms=hotel_rooms.dropna()
hotel_details.drop_duplicates(subset='hotelid',keep=False,inplace=True)
hotel=pd.merge(hotel_rooms,hotel_details,left_on='hotelcode',right_on='hotelid',how='inner')
hotel.columns
del hotel['hotelid']

del hotel['url']

del hotel['curr']

del hotel['Source']
hotel.columns
def citybased(city):

    hotel['city']=hotel['city'].str.lower()

    citybase=hotel[hotel['city']==city.lower()]

    citybase=citybase.sort_values(by='starrating',ascending=False)

    citybase.drop_duplicates(subset='hotelcode',keep='first',inplace=True)

    if(citybase.empty==0):

        hname=citybase[['hotelname','starrating','address','roomamenities','ratedescription']]

        return hname.head()

    else:

        print('No Hotels Available')
print('Top 5 hotels')

citybased('London')
room_no=[

     ('king',2),

   ('queen',2), 

    ('triple',3),

    ('master',3),

   ('family',4),

   ('murphy',2),

   ('quad',4),

   ('double-double',4),

   ('mini',2),

   ('studio',1),

    ('junior',2),

   ('apartment',4),

    ('double',2),

   ('twin',2),

   ('double-twin',4),

   ('single',1),

     ('diabled',1),

   ('accessible',1),

    ('suite',2),

    ('one',2)

   ]
def calc():

    guests_no=[]

    for i in range(hotel.shape[0]):

        temp=hotel['roomtype'][i].lower().split()

        flag=0

        for j in range(len(temp)):

            for k in range(len(room_no)):

                if temp[j]==room_no[k][0]:

                    guests_no.append(room_no[k][1])

                    flag=1

                    break

            if flag==1:

                break

        if flag==0:

            guests_no.append(2)

    hotel['guests_no']=guests_no



calc()
def pop_citybased(city,number):

    hotel['city']=hotel['city'].str.lower()

    popbased=hotel[hotel['city']==city.lower()]

    popbased=popbased[popbased['guests_no']==number].sort_values(by='starrating',ascending=False)

    popbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)

    if popbased.empty==True:

        print('Sorry No Hotels Available\n tune your constraints')

    else:

        return popbased[['hotelname','roomtype','guests_no','starrating','address','roomamenities','ratedescription']].head(10)

    

    
pop_citybased('London',4)
hotel['roomamenities']=hotel['roomamenities'].str.replace(': ;',',')
def requirementbased(city,number,features):

    hotel['city']=hotel['city'].str.lower()

    hotel['roomamenities']=hotel['roomamenities'].str.lower()

    features=features.lower()

    features_tokens=word_tokenize(features)  

    sw = stopwords.words('english')

    lemm = WordNetLemmatizer()

    f1_set = {w for w in features_tokens if not w in sw}

    f_set=set()

    for se in f1_set:

        f_set.add(lemm.lemmatize(se))

    reqbased=hotel[hotel['city']==city.lower()]

    reqbased=reqbased[reqbased['guests_no']==number]

    reqbased=reqbased.set_index(np.arange(reqbased.shape[0]))

    l1 =[];l2 =[];cos=[];

    #print(reqbased['roomamenities'])

    for i in range(reqbased.shape[0]):

        temp_tokens=word_tokenize(reqbased['roomamenities'][i])

        temp1_set={w for w in temp_tokens if not w in sw}

        temp_set=set()

        for se in temp1_set:

            temp_set.add(lemm.lemmatize(se))

        rvector = temp_set.intersection(f_set)

        #print(rvector)

        cos.append(len(rvector))

    reqbased['similarity']=cos

    reqbased=reqbased.sort_values(by='similarity',ascending=False)

    reqbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)

    return reqbased[['hotelname','roomtype','guests_no','starrating','address','roomamenities','ratedescription','similarity']].head(10)
requirementbased('London',1,'I need a extra toilet and room should be completely air conditioned.I should have a bathrobe.')
from math import sin, cos, sqrt, atan2, radians

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_code")

import requests

R=6373.0#Earth's Radius

sw = stopwords.words('english')

lemm = WordNetLemmatizer()

def hybrid(address,city,number,features):

    features=features.lower()

    features_tokens=word_tokenize(features)

    f1_set = {w for w in features_tokens if not w in sw}

    f_set=set()

    for se in f1_set:

        f_set.add(lemm.lemmatize(se))

    data = {

    'key': secret_value_0,

    'q': address,

    'format': 'json'}

    response = requests.get(url, params=data)

    dist=[]

    lat1,long1=response.json()[0]['lat'],response.json()[0]['lon']

    lat1=radians(float(lat1))

    long1=radians(float(long1))

    hybridbase=hotel[hotel['guests_no']==number]

    hybridbase['city']=hybridbase['city'].str.lower()

    hybridbase=hybridbase[hybridbase['city']==city.lower()]

    hybridbase.drop_duplicates(subset='hotelcode',inplace=True,keep='first')

    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))

    for i in range(hybridbase.shape[0]):

        lat2=radians(hybridbase['latitude'][i])

        long2=radians(hybridbase['longitude'][i])

        dlon = long2 - long1

        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        dist.append(distance)

    hybridbase['distance']=dist

    hybridbase=hybridbase.sort_values(by='distance',ascending=True)

    hybridbase=hybridbase.head(15)

    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))

    coss=[]

    for i in range(hybridbase.shape[0]):

        temp_tokens=word_tokenize(hybridbase['roomamenities'][i])

        temp1_set={w for w in temp_tokens if not w in sw}

        temp_set=set()

        for se in temp1_set:

            temp_set.add(lemm.lemmatize(se))

        rvector = temp_set.intersection(f_set)

        coss.append(len(rvector))

    hybridbase['similarity']=coss

    return hybridbase.sort_values(by='similarity',ascending=False).head(10)

    

    

        
url = "https://us1.locationiq.com/v1/search.php"

hybrid("Big Ben,London",'London',4,'I need a extra toilet and room should be completely air conditioned.I should have a bathrobe.')
hotel_cost.head()

#onsite rate is one important feature which could be useful to recommend

#we will drop the rest since it is present in other table and we are going to merge the 

hotel_cost=hotel_cost.drop(['id','refid','websitecode','dtcollected','ratedate','los','guests','roomtype','netrate','ratedescription','ratetype','sourceurl','roomamenities'

,'ispromo','closed','discount','promoname','status_code','taxstatus','taxtype','taxamount','proxyused','israteperstay','hotelblock','input_dtcollected'],axis=1)
hotel_cost.columns
#To reccomend we are gonna check how much does the price vary from room to room if 

#the varience is small enough then it is better for them to recommend the hotel

hot=hotel_cost.groupby(['hotelcode','maxoccupancy'])
hotel_cost.sort_values(by=['onsiterate'],ascending=False)

hotel_cost=hotel_cost.drop_duplicates(subset=['hotelcode','maxoccupancy'],keep='first')
var=hot['onsiterate'].var().to_frame('varience')

l=[]

for i in range(hotel_cost.shape[0]):

    var1=var[var.index.get_level_values(0)==hotel_cost.iloc[i][0]]

    l.append(var1[var1.index.get_level_values(1)==hotel_cost.iloc[i][3]]['varience'][0])
hotel_cost['var']=l

hotel_cost=hotel_cost.fillna(0)

hotel_cost['mealinclusiontype']=hotel_cost['mealinclusiontype'].replace(0,'No Complimentary')
hotel1=pd.merge(hotel,hotel_cost,left_on=['hotelcode','guests_no'],right_on=['hotelcode','maxoccupancy'],how='inner')
hotel1=hotel1.drop_duplicates(subset=['hotelcode','maxoccupancy'],keep='first')
hotel1.columns
def pricing(address,city,number,features):

    h=hybrid(address,city,number,features)

    price_based=pd.merge(h,hotel_cost,left_on=['hotelcode','guests_no'],right_on=['hotelcode','maxoccupancy'],how='inner')

    del price_based['maxoccupancy']

    h=price_based.sort_values(by='var')

    return h.head()
pricing("Tower of London",'London',4,'I need an alarm clock and a kettle flask.')

#the description is just only for example.It portrays you can keep any specific request
optimum_band=pd.read_csv('../input/hotel-recommendation/hotel_price_min_max - Formula.csv')
from math import sin, cos, sqrt, atan2, radians

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_code")

import requests

R=6373.0#Earth's Radius

sw = stopwords.words('english')

lemm = WordNetLemmatizer()

def hybrid(address,city,number,features):

    features=features.lower()

    features_tokens=word_tokenize(features)

    f1_set = {w for w in features_tokens if not w in sw}

    f_set=set()

    for se in f1_set:

        f_set.add(lemm.lemmatize(se))

    data = {

    'key': secret_value_0,

    'q': address,

    'format': 'json'}

    response = requests.get(url, params=data)

    dist=[]

    lat1,long1=response.json()[0]['lat'],response.json()[0]['lon']

    lat1=radians(float(lat1))

    long1=radians(float(long1))

    hybridbase=hotel

    #hybridbase=hotel[hotel['guests_no']==number]

    hybridbase['city']=hybridbase['city'].str.lower()

    hybridbase=hybridbase[hybridbase['city']==city.lower()]

    hybridbase.drop_duplicates(subset='hotelcode',inplace=True,keep='first')

    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))

    for i in range(hybridbase.shape[0]):

        lat2=radians(hybridbase['latitude'][i])

        long2=radians(hybridbase['longitude'][i])

        dlon = long2 - long1

        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        dist.append(distance)

    hybridbase['distance']=dist

    hybridbase=hybridbase[hybridbase['distance']<=5]

    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))

    coss=[]

    for i in range(hybridbase.shape[0]):

        temp_tokens=word_tokenize(hybridbase['roomamenities'][i])

        temp1_set={w for w in temp_tokens if not w in sw}

        temp_set=set()

        for se in temp1_set:

            temp_set.add(lemm.lemmatize(se))

        rvector = temp_set.intersection(f_set)

        coss.append(len(rvector))

    hybridbase['similarity']=coss

    return hybridbase.sort_values(by='similarity',ascending=False)



def price_band_based(address,city,number,features):

    h=hybrid(address,city,number,features)

    price_band=pd.merge(h,optimum_band,left_on=['hotelcode'],right_on=['hotelcode'],how='inner')

    price_band=pd.merge(price_band,hotel_cost,left_on=['hotelcode'],right_on=['hotelcode'],how='inner')

    del price_band['min']

    del price_band['max']

    del price_band['Diff_Min']

    del price_band['Diff_Max']

    del price_band['currency']

    del price_band['country']

    del price_band['propertytype']

    del price_band['starrating']

    del price_band['latitude']

    del price_band['longitude']

    del price_band['guests_no']

    del price_band['var']

    price_band=price_band[price_band['Score']<=0.5]

    price_band=price_band[price_band['maxoccupancy']>=number]

    return price_band
price_band_based("Tower of London",'London',4,'I need an alarm clock and a kettle flask.').head(5)
import pickle
Pkl_Filename = "Pickle_RL_Model.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(price_band_based, file)