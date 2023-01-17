import pandas as pd

import seaborn as sns

import numpy as np

from scipy.stats import norm

import scipy

import matplotlib.pyplot as plt

from tqdm import tqdm

pd.set_option('display.max_columns', 500)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv('../input/listings_summary.csv')

df.head(2)
columns_to_keep = ['id','host_has_profile_pic','host_since','neighbourhood_cleansed', 'neighbourhood_group_cleansed',

                   'host_is_superhost','description',

                   'latitude', 'longitude','is_location_exact', 'property_type', 'room_type', 'accommodates', 'bathrooms',  

                   'bedrooms', 'bed_type', 'amenities', 'price', 'cleaning_fee',

                   'review_scores_rating','reviews_per_month','number_of_reviews',

                   'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',

                   'review_scores_communication','review_scores_location','review_scores_value',

                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  

                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy','availability_365']



df = df[columns_to_keep].set_index('id')

df.head(2)
df.isnull().sum()
df['is_location_exact'] = df['is_location_exact'].map({'f':0,'t':1})

df['host_is_superhost'] = df['host_is_superhost'].map({'f':0,'t':1})

df['is_business_travel_ready'] = df['is_business_travel_ready'].map({'f':0,'t':1})

df['instant_bookable'] = df['instant_bookable'].map({'f':0,'t':1})
df.head(3)
set(df['host_has_profile_pic'])
df['host_has_profile_pic'].fillna('f',inplace=True)
df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'f':0,'t':1})

sns.countplot(x='host_has_profile_pic',data=df)
df['host_has_profile_pic'].value_counts()
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)

df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)

df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',', '').astype(float)
df['cleaning_fee'].fillna(df['cleaning_fee'].median(), inplace=True)

df['cleaning_fee'].isna().sum()
df['security_deposit'].fillna(df['security_deposit'].median(), inplace=True)

df['security_deposit'].isna().sum()
df['price'].describe()
set1=set(i for i in df[(df['price']==0)].index.tolist())

len(set1)
df = df.drop(list(set1))

df.reset_index(inplace=True)

df['price'] = np.log1p(df['price'])
sns.distplot(df['price'], fit=norm);

fig = plt.figure()

res = scipy.stats.probplot(df['price'], plot=plt)

print("Skewness: %f" % df['price'].skew())

print("Kurtosis: %f" % df['price'].kurt())
sns.countplot(x='room_type',data=df)
sns.countplot(x='neighbourhood_group_cleansed',data=df)
sns.countplot(x='neighbourhood_cleansed',data=df)
z = df['neighbourhood_cleansed'].value_counts()
others = []

for i in set(df['neighbourhood_cleansed']):

    if z[i]<100:

        others.append(i)

len(others)
for i in tqdm(range(len(df))):

    if df.loc[i,'neighbourhood_cleansed'] in others:

        df.loc[i,'neighbourhood_cleansed'] = 'Others'
z = df['property_type'].value_counts()
others = []

for i in set(df['property_type']):

    if z[i]<100:

        others.append(i)

len(others)
for i in tqdm(range(len(df))):

    if df.loc[i,'neighbourhood_cleansed'] in others:

        df.loc[i,'neighbourhood_cleansed'] = 'Others'
df['bathrooms'].value_counts()
df['bathrooms'].fillna(1,inplace=True)
df['bedrooms'].value_counts()
df['bedrooms'].fillna(1,inplace=True)
type(list(set(df['host_since']))[0])
set2=[]

z = df['host_since'].isnull()

for i in range(len(z)):

    if z.loc[i]==True:

        set2.append(i)

z = df['host_is_superhost'].isnull()

for i in range(len(z)):

    if z.loc[i]==True:

        set2.append(i)

set2 = set(set2)

len(set2)
df = df.drop(list(set2))

df.reset_index(inplace=True)
dropped = ['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',

            'review_scores_communication','review_scores_location','review_scores_value']

df.drop(dropped,axis=1,inplace=True)
df.head(2)
df['cancellation_policy'].value_counts()
y = df['price']

df.drop(['price'],axis=1,inplace=True)
df.isnull().sum()
df.head(2)
df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)

df['size'] = df['size'].str.replace("\D", "")

df['size'] = df['size'].astype(float)

sub_df = df[['accommodates', 'bathrooms', 'bedrooms', 'cleaning_fee', 

                 'security_deposit', 'extra_people', 'guests_included', 'size']]
train_data = sub_df[sub_df['size'].notnull()]

test_data  = sub_df[sub_df['size'].isnull()]



X_train = train_data.drop('size', axis=1)

X_test  = test_data.drop('size', axis=1)



y_train = train_data['size']
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)
y_test = linreg.predict(X_test)
mse(y_train,linreg.predict(X_train)),r2_score(y_train,linreg.predict(X_train))

y_test = pd.DataFrame(y_test)

y_test.columns = ['size']
prelim_index = pd.DataFrame(X_test.index)

prelim_index.columns = ['prelim']



y_test = pd.concat([y_test, prelim_index], axis=1)

y_test.set_index(['prelim'], inplace=True)

new_test_data = pd.concat([X_test, y_test], axis=1)

sub_df_new = pd.concat([new_test_data, train_data], axis=0)
sub_df_new.columns
df.drop(['size'],axis=1,inplace=True)

sub_df_new = sub_df_new['size'] 
df = pd.concat([sub_df_new, df], axis=1)
df.head(3)
dropped = ['index','id','description']

df.drop(dropped,axis=1,inplace=True)
df.head(3)
df['No_of_amentities'] = df['amenities'].apply(lambda x:len(x.split(',')))
df.head(3)
df['Laptop_friendly_workspace'] = df['amenities'].str.contains('Laptop friendly workspace')

df['TV'] = df['amenities'].str.contains('TV')

df['Family_kid_friendly'] = df['amenities'].str.contains('Family/kid friendly')

df['Host_greets_you'] = df['amenities'].str.contains('Host greets you')

df['Smoking_allowed'] = df['amenities'].str.contains('Smoking allowed')

df['Hot_water'] = df['amenities'].str.contains('Hot water')

df['Fridge'] = df['amenities'].str.contains('Refrigerator')

df.head(2)
dropped = ['amenities']

df.drop(dropped,axis=1,inplace=True)
category = ['neighbourhood_cleansed','neighbourhood_group_cleansed','property_type','room_type',

           'bed_type','cancellation_policy']



for i in category:

    df[i] = df[i].astype('category')

    df[i] = df[i].cat.codes
df.head(3)
from dateutil import parser



def diff_date(row):

    today = parser.parse('2018-11-7')

    return ((today - parser.parse(row['host_since'])).days)/365.25

df['host_since'] = df.apply(diff_date,axis=1)

df['host_since'].describe()
from math import sin, cos, sqrt, atan2, radians
def haversine_distance_central(row):

    berlin_lat,berlin_long = radians(52.5200), radians(13.4050)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
def haversine_distance_airport(row):

    berlin_lat,berlin_long = radians(52.3733), radians(13.5064)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
def haversine_distance_rail(row):

    berlin_lat,berlin_long = radians(52.5073), radians(13.3324)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
df['distance_central'] = df.apply(haversine_distance_central,axis=1)

df['distance_airport'] = df.apply(haversine_distance_airport,axis=1)

df['distance_railways'] = df.apply(haversine_distance_airport,axis=1)

df['distance_avg'] = ( df['distance_central'] + df['distance_airport'] + df['distance_railways'] )/3.0
df['distance_avg'].describe()
df.head()
for c in category:

    df[c+'_freq'] = df[c].map(df.groupby(c).size() / df.shape[0])

    indexer = pd.factorize(df[c], sort=True)[1]

    df[c] = indexer.get_indexer(df[c])
df.head(3)
df.isna().sum()
df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())
df.to_csv('X_new.csv',index=False)
y.to_csv('Y_new.csv',index=False)
df = pd.read_csv('X_new.csv')

df.head(2)
y = pd.read_csv('Y_new.csv',header=None)

y.head(2)
X_train, X_val , y_train, y_val = train_test_split(df,y,test_size=0.3)
scaler = StandardScaler()

scaler.fit_transform(X_train)

scaler.transform(X_val)
def adj_r2(r2,n,p):

    return 1- ((1-r2)*(n-1))/(n-p-1)
model = Lasso(alpha=1e-6)
model.fit(X_train,y_train)
yp_train = model.predict(X_train)

yp_val = model.predict(X_val)
train_r2 =(r2_score(y_train, yp_train))

val_r2 =(r2_score(y_val, yp_val))

    

print('Train r2= ',train_r2)

print('Test r2= ',val_r2)

train_mse =(mse(y_train, yp_train))

val_mse =(mse(y_val, yp_val))

    

print('Train error= ',train_mse)

print('Test error= ',val_mse)

adj_r2(train_r2,X_train.shape[0],X_train.shape[1])
adj_r2(val_r2,X_val.shape[0],X_val.shape[1])
for i in range(len(model.coef_)):

    print(df.columns[i],' ',model.coef_[i])
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=5,verbose=1,random_state=0,n_estimators=200,n_jobs=-1)
model.fit(X_train,y_train)
yp_train = model.predict(X_train)

yp_val = model.predict(X_val)
train_r2 =(r2_score(y_train, yp_train))

val_r2 =(r2_score(y_val, yp_val))

    

print('Train r2= ',train_r2)

print('Test r2= ',val_r2)
train_mse =(mse(y_train, yp_train))

val_mse =(mse(y_val, yp_val))

    

print('Train error= ',train_mse)

print('Test error= ',val_mse)
adj_r2(train_r2,X_train.shape[0],X_train.shape[1])
adj_r2(val_r2,X_val.shape[0],X_val.shape[1])
for i in range(len(X_train.columns)):

    print(i,' ',X_train.columns[i])
import lightgbm as lgbm

params = {'objective': 'regression',

          'metric': 'rmse',

          'learning_rate':0.005,

          'max_depth':6

         } 

train_set = lgbm.Dataset(X_train,y_train, silent=True)

model = lgbm.train(params, train_set=train_set,num_boost_round=1000,categorical_feature=[9,10,15,16,20,30])
yp_train = model.predict(X_train)

yp_val = model.predict(X_val)
train_r2 =(r2_score(y_train, yp_train))

val_r2 =(r2_score(y_val, yp_val))

    

print('Train r2= ',train_r2)

print('Test r2= ',val_r2)

train_mse =(mse(y_train, yp_train))

val_mse =(mse(y_val, yp_val))

    

print('Train error= ',train_mse)

print('Test error= ',val_mse)

adj_r2(train_r2,X_train.shape[0],X_train.shape[1])
adj_r2(val_r2,X_val.shape[0],X_val.shape[1])