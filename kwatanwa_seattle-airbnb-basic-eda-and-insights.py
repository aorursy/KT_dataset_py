# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
listings_df = pd.read_csv("/kaggle/input/seattle/listings.csv")

listings_df[:5]
calender_df = pd.read_csv("/kaggle/input/seattle/calendar.csv")

calender_df[:5]
reviews_df = pd.read_csv("/kaggle/input/seattle/reviews.csv")

reviews_df[:5]
# Define a useful function

def disp_freq(df, col):

    """ Display a bar plot with counts

    

    Arguments:

        df(dataframe): an arbitrary dataframe

        col(str): one column of the dataframe

    

    """

    return df[col].value_counts().plot(kind='bar', title = col +' counts');
# Data type counts

listings_df.dtypes.value_counts()
# Column names

list(listings_df.columns.values)
# Float data type columns

listings_df.select_dtypes(include = ["float"])[:10]
# Integer data type columns

listings_df.select_dtypes(include = ["int"])[:5]
# Object data type columns

listings_df.select_dtypes(include = ["object"])[:5]
# Object data type column names

list(listings_df.select_dtypes(include = ["object"]).columns.values)
# NA counts

listings_df.isna().sum()
# Which clumns have missig values ?

listings_df.isna().sum()[listings_df.isna().sum() > 0]
# Price cols

price_cols = [col for col in listings_df.columns if 'price' in col]

price_cols
# See the columns which contain 'price'

listings_df[price_cols][:5]
# Location related columns

listings_df[['city','state','zipcode','market','smart_location','country_code','country']].describe()
# See location variations grouped by 'state'

listings_df[['city','state','zipcode','smart_location']].groupby('state').count()
# See location variations grouped by 'smart_location'

listings_df[['city','zipcode','smart_location']].groupby('smart_location').count()
# See location variations grouped by 'city'

listings_df[['city','zipcode']].groupby('city').count()
# Zipcode counts

disp_freq(listings_df, 'zipcode');

# Zipcode null counts

listings_df['zipcode'].isna().sum()
# Review related columns

review_cols = [col for col in listings_df.columns if 'review' in col]

listings_review_df = listings_df[review_cols]

listings_review_df
# number_of_reviews

listings_review_df['number_of_reviews'].value_counts()
# Plot

listings_review_df['number_of_reviews'].value_counts().hist();
listings_review_description = listings_review_df['number_of_reviews'].describe().to_frame()
# Use for the blog post

fig, ax = plt.subplots(figsize=(5,5))

ax.axis('off')

ax.table(cellText = listings_review_description.values, rowLabels = listings_review_description.index, loc='center', bbox=[1,0,1,1]);
# Review euals 10 or less

np.sum(listings_review_df['number_of_reviews'] < 11)
# Review euals 5 or less

np.sum(listings_review_df['number_of_reviews'] < 6)
# Review euals 3 or less

np.sum(listings_review_df['number_of_reviews'] < 4)
listings_df[['price','weekly_price','monthly_price','security_deposit','cleaning_fee']]
# Change data type

listings_price_df = listings_df[['price','weekly_price','monthly_price','security_deposit','cleaning_fee']]



for col in ['price','weekly_price','monthly_price','security_deposit','cleaning_fee']:

    listings_price_df = pd.concat([listings_price_df.drop(columns = [col]), listings_price_df[col].str.replace('$','').str.replace(',','').astype(float)], axis = 1)

    

listings_price_df[:5]
# NA counts

listings_price_df.isna().sum()/listings_price_df.shape[0]
# calculate weekly and monthly prices

listings_price_df['calc_weekly_price'] = listings_price_df['price'] * 7

listings_price_df['calc_monthly_price'] = listings_price_df['price'] * 30

listings_price_df[:5]
# Fill missing values as 0

listings_price_df.fillna(0, inplace = True)

listings_price_df[:5]
# loop

for idx, row in listings_price_df.iterrows():

    if row['weekly_price'] == 0:

        listings_price_df.loc[idx, ['weekly_price']] = row['calc_weekly_price']

    if row['monthly_price'] == 0:

        listings_price_df.loc[idx, ['monthly_price']] = row['calc_monthly_price']



listings_price_df[:5]
# Drop calc_weekly_price and calc_monthly_price columns

listings_price_df = listings_price_df.drop(columns = ['calc_weekly_price', 'calc_monthly_price'])
listings_price_df[:5]
# security_deposit

listings_price_df[['security_deposit', 'cleaning_fee']].hist(figsize = (10, 5));
listings_price_df['security_deposit'].describe()
listings_price_df['cleaning_fee'].plot(kind='bar');
listings_price_df['cleaning_fee'].hist();
listings_price_df['cleaning_fee'].describe()
# Correlation

sns.heatmap(listings_price_df.corr(), annot = True);
review_scores_cols = [col for col in listings_df.columns if 'review_scores' in col]

review_scores_df = listings_df[review_scores_cols]
# Missing values

review_scores_df.isna().sum()
# drop na rows

review_scores_df = review_scores_df.dropna()

review_scores_df.describe()
# review_scores_rating

review_scores_df['review_scores_rating'].plot(kind='hist');
# Other scores

review_scores_df.hist(layout = [4,2], figsize = (10, 10), bins = 10);
# Correlations

sns.heatmap(review_scores_df.corr(method ='pearson'), annot = True);
facilities_df = listings_df[['bathrooms', 'bedrooms', 'beds', 'accommodates']]

facilities_df[:5]
#Missing values

facilities_df.isna().sum()
facilities_df[facilities_df.isna().any(axis=1)]
listings_df['accommodates'].value_counts().sort_index()
listings_df['accommodates'].value_counts().sort_index().plot(kind='bar');
listings_df[['property_type','room_type','bed_type','amenities']][:5]
# NA counts

listings_df[['property_type','room_type','bed_type','amenities']].isna().sum()
# Na rows

(listings_df[['property_type','room_type','bed_type','amenities']])[listings_df[['property_type','room_type','bed_type','amenities']].isna().any(axis=1)]
property_type_df = listings_df['property_type'].value_counts().reset_index()

property_type_df.columns = ['Property type','Count']

property_type_df.plot(x = 'Property type', y = 'Count', kind = 'bar');
room_type_df = listings_df['room_type'].value_counts().reset_index()

room_type_df.columns = ['Room type','Count']

room_type_df.plot(x = 'Room type', y = 'Count', kind = 'bar');
bed_type_df = listings_df['bed_type'].value_counts().reset_index()

bed_type_df.columns = ['Bed type','Count']

bed_type_df.plot(x = 'Bed type', y = 'Count', kind = 'bar');
# amenities

amenities_df = listings_df['amenities']

amenities_df[:5]
amenities_df.value_counts()
amenities_df = amenities_df[amenities_df != '{}']
amenities_list = []



for index, row in amenities_df.items():

    amenities_list.append(row.replace('{','').replace('}','').replace('"','').split(','))



amenities_list[:3]

# Create a new ammenities df



new_amenities_df = pd.Series(amenities_list, name = 'amenities').to_frame()

new_amenities_df
# Modified the code from https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list



dummies_amenities_df = new_amenities_df.drop('amenities', 1).join(

    pd.get_dummies(

        pd.DataFrame(new_amenities_df.amenities.tolist()).stack()

    ).astype(int).sum(level=0)

)



dummies_amenities_df
dummies_amenities_df.columns.values
dummies_amenities_df.sum()
# Wich row where Washer / Dryer is one

dummies_amenities_df[dummies_amenities_df['Washer / Dryer'] == 1]
dummies_amenities_df = dummies_amenities_df.drop(columns = ['Washer / Dryer'])
dummies_amenities_df.sum().sort_values(ascending = False).plot(kind='bar', figsize = (15,5));
policies_df = listings_df[['cancellation_policy',

 'require_guest_profile_picture',

 'require_guest_phone_verification']]



policies_df[:5]
# cancellation policy



policies_df['cancellation_policy'].value_counts()
policies_df['require_guest_profile_picture'].value_counts()
policies_df['require_guest_phone_verification'].value_counts()
# cancellation_policy dummy df

pd.get_dummies(policies_df['cancellation_policy'], prefix='cancellation_policy')[:5]
# See first 5 rows

calender_df[:5]
# data types

calender_df.dtypes
# Null counts



calender_df.isna().sum()/calender_df.shape[0]
# available variable counts



calender_df['available'].value_counts()/calender_df.shape[0]
# listing_id counts

calender_df['listing_id'].value_counts()
# Unique listing_id

calender_df['listing_id'].value_counts().shape[0]
# Number of date unique values

calender_df['date'].value_counts().shape[0]
# Date variable counts

calender_df['price'].value_counts()
#Select only the columns which we are interested in



selected_listings_cols = [

    'id',

#  'listing_url',

#  'scrape_id',

#  'last_scraped',

#  'name',

#  'summary',

#  'space',

#  'description',

#  'experiences_offered',

#  'neighborhood_overview',

#  'notes',

#  'transit',

#  'thumbnail_url',

#  'medium_url',

#  'picture_url',

#  'xl_picture_url',

#  'host_id',

#  'host_url',

#  'host_name',

#  'host_since',

#  'host_location',

#  'host_about',

#  'host_response_time',

#  'host_response_rate',

#  'host_acceptance_rate',

#  'host_is_superhost',

#  'host_thumbnail_url',

#  'host_picture_url',

#  'host_neighbourhood',

#  'host_listings_count',

#  'host_total_listings_count',

#  'host_verifications',

#  'host_has_profile_pic',

#  'host_identity_verified',

#  'street',

#  'neighbourhood',

#  'neighbourhood_cleansed',

#  'neighbourhood_group_cleansed',

#  'city',

#  'state',

 'zipcode',

#  'market',

#  'smart_location',

#  'country_code',

#  'country',

#  'latitude',

#  'longitude',

#  'is_location_exact',

 'property_type',

 'room_type',

 'accommodates',

 'bathrooms',

 'bedrooms',

 'beds',

 'bed_type',

 'amenities',

#  'square_feet',

 'price',

 'weekly_price',

 'monthly_price',

 'security_deposit',

 'cleaning_fee',

#  'guests_included',

#  'extra_people',

#  'minimum_nights',

#  'maximum_nights',

#  'calendar_updated',

#  'has_availability',

#  'availability_30',

#  'availability_60',

#  'availability_90',

#  'availability_365',

#  'calendar_last_scraped',

 'number_of_reviews',

#  'first_review',

#  'last_review',

 'review_scores_rating',

 'review_scores_accuracy',

 'review_scores_cleanliness',

 'review_scores_checkin',

 'review_scores_communication',

 'review_scores_location',

 'review_scores_value',

#  'requires_license',

#  'license',

#  'jurisdiction_names',

#  'instant_bookable',

 'cancellation_policy',

#  'require_guest_profile_picture',

#  'require_guest_phone_verification',

#  'calculated_host_listings_count',

#  'reviews_per_month'

]



new_listings_df = listings_df[selected_listings_cols]

new_listings_df[:5]
# Drop rows with missing values of the number of reviews

# Drop rows with missing values of the bathrooms

# Drop rows with missing values of the property type



review_scores_cols = [col for col in listings_df.columns if 'review_scores' in col]

drop_cols = ['number_of_reviews','bathrooms','property_type'] + review_scores_cols



new_listings_df.dropna(subset = drop_cols, axis = 0, inplace = True)



# Drop rows where the number of reviews is 0.

new_listings_df = new_listings_df[new_listings_df['number_of_reviews'] != 0]



# Drop rows where the amenities are empty

new_listings_df = new_listings_df[new_listings_df['amenities'] != '{}']



# Reset index

new_listings_df = new_listings_df.reset_index(drop = True)
# Fill missing values of the bedrooms and beds as 1

new_listings_df[['bedrooms', 'beds']] = new_listings_df[['bedrooms', 'beds']].fillna(value = 1)
# Change the data type of the price and related columns and fill missing values as 0

new_listings_price_df = new_listings_df[['price','weekly_price','monthly_price','security_deposit','cleaning_fee']]



for col in ['price','weekly_price','monthly_price','security_deposit','cleaning_fee']:

    new_listings_price_df = pd.concat([new_listings_price_df.drop(columns = [col]), new_listings_price_df[col].str.replace('$','').str.replace(',','').astype(float)], axis = 1)

    

new_listings_price_df.fillna(value = 0, inplace =True)
# Calculate weekly and monthly price

new_listings_price_df['calc_weekly_price'] = new_listings_price_df['price'] * 7

new_listings_price_df['calc_monthly_price'] = new_listings_price_df['price'] * 30

new_listings_price_df[:5]
# Fill the weekly and monthky price by its calculated values

for idx, row in new_listings_price_df.iterrows():

    if row['weekly_price'] == 0:

        new_listings_price_df.loc[idx, ['weekly_price']] = row['calc_weekly_price']

    if row['monthly_price'] == 0:

        new_listings_price_df.loc[idx, ['monthly_price']] = row['calc_monthly_price']



new_listings_price_df[:5]
# Drop calculated weekly and monthly price columns

new_listings_price_df.drop(columns = ['calc_weekly_price', 'calc_monthly_price'], inplace = True)



# # Reset index

# new_listings_price_df = new_listings_price_df.reset_index(drop=True)
# Create dummy columns of cancellation policy, room type, property type and bed type



cancellation_policy_dummy_df = pd.get_dummies(new_listings_df['cancellation_policy'], prefix = 'cancellation_policy')

room_type_dummy_df = pd.get_dummies(new_listings_df['room_type'], prefix = 'room_type')

property_type_dummy_df = pd.get_dummies(new_listings_df['property_type'], prefix = 'property_type')

bed_type_dummy_df = pd.get_dummies(new_listings_df['bed_type'], prefix = 'bed_type')

bed_type_dummy_df
# Create dummy columns based on the ammenities



# Drop rows with empty rows

amenities_series = new_listings_df['amenities']

amenities_series = amenities_series[amenities_series != '{}']



# Iterate over rows and format them as list

amenities_list = []



for index, row in amenities_series.items():

    amenities_list.append(row.replace('{','').replace('}','').replace('"','').split(','))



# Convert the list to a data frame

amenities_df = pd.Series(amenities_list, name = 'amenities').to_frame()



# Create a dummy data frame

dummies_amenities_df = amenities_df.drop('amenities', 1).join(

    pd.get_dummies(

        pd.DataFrame(amenities_df.amenities.tolist()).stack()

    ).astype(int).sum(level=0)

)



# Reset index

# dummies_amenities_df = dummies_amenities_df.reset_index(drop=True)

dummies_amenities_df[:5]
dummy_df = pd.concat([cancellation_policy_dummy_df, room_type_dummy_df, property_type_dummy_df, bed_type_dummy_df, dummies_amenities_df], axis = 1)

dummy_df[:5]
# Numeric columns

new_listings_df.select_dtypes(include = ['int', 'float'])
concat_listings_df = pd.concat([new_listings_df.select_dtypes(include = ['int', 'float']), new_listings_price_df, dummy_df], axis = 1)
concat_listings_df[:5]
# Check missing values

concat_listings_df.isna().sum().sum()
# Copy the original data set

new_calender_df = calender_df.copy()

new_calender_df
# Change data types

new_calender_df['date'] = pd.to_datetime(new_calender_df['date']) 

new_calender_df['price'] = new_calender_df['price'].str.replace('$','').str.replace(',','').astype(float)

new_calender_df['available'] = new_calender_df['available'].replace({'t': True, 'f': False})

new_calender_df
# Plot price data

new_calender_df['price'].plot();
# Na counts by id

new_calender_df.groupby('listing_id')['available'].sum()
new_calender_df.groupby('listing_id')['available'].sum().max()
# How many rooms or apartments are never available ?

np.sum(new_calender_df.groupby('listing_id')['available'].sum() == 0)
# Filter out never available listing_id

listing_id_available_count = new_calender_df.groupby('listing_id')['available'].sum().loc[lambda x : x != 0]
listing_id_available_count.plot(kind='hist');
# Number of always available listing_id

listing_id_available_count[listing_id_available_count == 365].count()
# Create a list which contains always non available listing_id

always_f_listing_id = list(new_calender_df.groupby('listing_id')['available'].sum().loc[lambda x : x == 0].index.values)
# Drop these rows

clean_calender_df = new_calender_df[~new_calender_df['listing_id'].isin(always_f_listing_id)]

np.sum(clean_calender_df.groupby('listing_id')['available'].sum() == 0)
# Create new columns

clean_calender_df['day'] = clean_calender_df['date'].dt.day

clean_calender_df['month'] = clean_calender_df['date'].dt.month

clean_calender_df['year'] = clean_calender_df['date'].dt.year
# Take a look at the cleaned data

clean_calender_df[:5]
# Check missing values

clean_calender_df.isna().sum()
# Use concat_listings_df

list(concat_listings_df.columns.values)
# Data preparation 



selected_cols = [

#     'id',

 'accommodates',

 'bathrooms',

 'bedrooms',

 'beds',

#  'number_of_reviews',

#  'review_scores_rating',

 'review_scores_accuracy',

 'review_scores_cleanliness',

 'review_scores_checkin',

 'review_scores_communication',

 'review_scores_location',

#  'review_scores_value',

#  'price',

#  'weekly_price',

#  'monthly_price',

#  'security_deposit',

#  'cleaning_fee',

 'cancellation_policy_flexible',

 'cancellation_policy_moderate',

 'cancellation_policy_strict',

 'room_type_Entire home/apt',

 'room_type_Private room',

 'room_type_Shared room',

 'property_type_Apartment',

 'property_type_Bed & Breakfast',

 'property_type_Boat',

 'property_type_Bungalow',

 'property_type_Cabin',

 'property_type_Camper/RV',

 'property_type_Chalet',

 'property_type_Condominium',

 'property_type_Dorm',

 'property_type_House',

 'property_type_Loft',

 'property_type_Other',

 'property_type_Tent',

 'property_type_Townhouse',

 'property_type_Treehouse',

 'property_type_Yurt',

 'bed_type_Airbed',

 'bed_type_Couch',

 'bed_type_Futon',

 'bed_type_Pull-out Sofa',

 'bed_type_Real Bed',

 '24-Hour Check-in',

 'Air Conditioning',

 'Breakfast',

 'Buzzer/Wireless Intercom',

 'Cable TV',

 'Carbon Monoxide Detector',

 'Cat(s)',

 'Dog(s)',

 'Doorman',

 'Dryer',

 'Elevator in Building',

 'Essentials',

 'Family/Kid Friendly',

 'Fire Extinguisher',

 'First Aid Kit',

 'Free Parking on Premises',

 'Gym',

 'Hair Dryer',

 'Hangers',

 'Heating',

 'Hot Tub',

 'Indoor Fireplace',

 'Internet',

 'Iron',

 'Kitchen',

 'Laptop Friendly Workspace',

 'Lock on Bedroom Door',

 'Other pet(s)',

 'Pets Allowed',

 'Pets live on this property',

 'Pool',

 'Safety Card',

 'Shampoo',

 'Smoke Detector',

 'Smoking Allowed',

 'Suitable for Events',

 'TV',

 'Washer',

#  'Washer / Dryer',

 'Wheelchair Accessible',

 'Wireless Internet'

                ]



# Exclude related columns and id

X = concat_listings_df[selected_cols]



y = concat_listings_df['price']
# Import required packages

from scipy.stats import uniform, randint

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.metrics import r2_score, mean_squared_error
# Train, test and validation data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=99) 
# Fit 

lm_model = LinearRegression(normalize=True) 

lm_model.fit(X_train, y_train)


#Predict and score the model

y_test_preds = lm_model.predict(X_test) 

y_train_preds = lm_model.predict(X_train) 



# r2 score

train_score = r2_score(y_train, y_train_preds)

test_score =  r2_score(y_test, y_test_preds)

print(train_score, test_score)
from xgboost import XGBRegressor

from xgboost import plot_importance
# Train, test and validation data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=99) 
# Create a model and fit the data to it 



xgb_model = XGBRegressor(

    max_depth=15,

    n_estimators=100,

    min_child_weight=10, 

    colsample_bytree=0.6, 

    subsample=0.6, 

    eta=0.2,    

    seed=0,

    learning_rate = 0.1,

    n_jobs=-1)



xgb_model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_val, y_val)], 

    verbose=10, 

    early_stopping_rounds = 10)

xgb_train_pred = xgb_model.predict(X_train)

xgb_val_pred = xgb_model.predict(X_val)

xgb_test_pred = xgb_model.predict(X_test)

for i in [[y_train, xgb_train_pred], [y_val, xgb_val_pred], [y_test, xgb_test_pred]]:

    print(r2_score(i[0], i[1]))
fig, ax = plt.subplots(figsize=(10, 15))

plot_importance(xgb_model,ax=ax);
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0, n_jobs=-1)

rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)

rf_val_pred = rf_model.predict(X_val)

rf_test_pred = rf_model.predict(X_test)
for i in [[y_train, rf_train_pred], [y_val, rf_val_pred], [y_test, rf_test_pred]]:

    print(r2_score(i[0], i[1]))
first_level = pd.DataFrame(xgb_val_pred, columns=["xgb"])

first_level["rf"] = rf_val_pred

first_level.info()



first_level_test = pd.DataFrame(xgb_test_pred, columns=["xgb"])

first_level_test["rf"] = rf_test_pred

first_level_test.info()
meta_model = LinearRegression(n_jobs=-1)

meta_model.fit(first_level, y_val)
test_prediction = meta_model.predict(first_level_test)
r2_score(y_test, test_prediction)
clean_calender_df[:5]
# Add zip code

# concat_listings_df['zipcode']
count_calender_df = clean_calender_df.groupby(['listing_id', 'year', 'month']).sum()['available'].reset_index()
count_calender_df['listing_id'] = count_calender_df.listing_id.astype(str)

count_calender_df[:20]
#  Drop data of 2017

count_calender_df = count_calender_df[count_calender_df['year'] != 2017]



# Drop the year column

count_calender_df = count_calender_df.drop(columns = ['year'])
max_month_availability = count_calender_df.groupby('month').max()['available']
max_month_availability
# Create a new column

count_calender_df['is_available'] = count_calender_df['available'] != 0
# Count the number of available rooms and apartments by month
count_calender_df.groupby('month').sum()['is_available'].plot();
# Only take rows where the available is True



price_tendency_df = clean_calender_df[clean_calender_df['available'] == True]
#  Drop data of 2017

price_tendency_df = price_tendency_df[price_tendency_df['year'] != 2017]



# Drop the year column

price_tendency_df = price_tendency_df.drop(columns = ['year'])
# Group by month

month_tendency_df = price_tendency_df.groupby('month').agg({"price": ["mean", 'median', 'min', 'max', 'std']}).reset_index(level=0)

month_tendency_df
month_tendency_df['price']['mean'].plot();