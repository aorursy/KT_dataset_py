import pandas as pd # create DataFrames
import numpy as np # working with arrays
import datetime # convert serial dates to datetime object
raw_data = pd.read_csv('../input/airbnb_nyc_homes_raw_20180908.csv') # read raw csv file
[raw_data.columns.values[x] for x in [43,87,88]] # retrieve columns that raised a warning
raw_data.info() #metadata
pd.set_option('display.max_columns', 96) # enlarge display of DataFrame to show more columns
raw_data.head(3) # head of DataFrame
raw_data.isna().sum()[raw_data.isna().sum() != 0] # columns that hold missing values
data = raw_data[[
'host_id',
'host_since',
'host_is_superhost',
'neighbourhood_cleansed',
'neighbourhood_group_cleansed',
'zipcode',
'latitude',
'longitude',
'room_type',
'bathrooms',
'bedrooms',
'beds',
'price',
'weekly_price',
'monthly_price',
'security_deposit',
'cleaning_fee',
'number_of_reviews',
'review_scores_rating',
'reviews_per_month',
]].copy().reset_index() # extracted columns to new DataFrame

data = data.rename(columns={'index':'id', 
                            'host_is_superhost':'superhost', 
                            'neighbourhood_cleansed':'neighborhood', 
                            'neighbourhood_group_cleansed': 'city'})
 # update column names
data.dtypes # retrieve datatypes
data.zipcode = pd.to_numeric(data.zipcode, errors='coerce',
                             downcast='integer') 
# changing zipcode to numeric. invalid parsing will be set as NaN.
data.isna().sum()[data.isna().sum() != 0] 
# retrieve all columns that have missing values
data = data.dropna(subset=['host_since','superhost','zipcode']) 
# create clean dataset
def serial_to_datetime(sdate):
# conversion formula
    temp = datetime.datetime(1900, 1, 1)
    delta = datetime.timedelta(days=sdate)
    return temp+delta

data['host_since'] = data['host_since'].apply(serial_to_datetime) 
# apply to all rows in host since column
data.zipcode = data.zipcode.astype(int) # change zip to an integer
data['city'] = data['city'].apply(
    lambda x: "New York" if x == "Manhattan" else x) 
# change "Manhattan" to "New York"
data.zipcode.loc[41368] = 11208 # reassign this zipcode because of a data entry order. Cypress Hill neighbourhood zip is 11208

data.zipcode.loc[30750] = 11367 # Kew Gardens Hills zipcode should be 11367, not 91766

data.zipcode.loc[35739] = 10036 # Hell's Kitchen zipcode should 10036, not 7093

data.zipcode.loc[[48316,48654]] = 10303 # Port Ivory zipcode should be 10303 not 7206

data.zipcode.loc[25165] = 10013 # SoHo zipcode should 10013, not 5340

data.zipcode.loc[39882] = 11207 # East New York zipcode should 11207, not 11954

data.neighborhood.loc[36027] = 'Howard Beach' # update to correct city
data.to_csv('airbnb_nyc_cleaned_20180908.csv', encoding='utf-8', sep=',', index=False) 
# save cleaned dataset csv
data[data['neighborhood'].isin(
    ['Westerleigh', 'Fort Wadsworth', 'Woodrow', 'Todt Hill', 'Tribeca'])].groupby(
    ['neighborhood']).agg({'price': "median", 'id':'count'}).rename(
    columns={'id': 'count of homes', 'price':'median price'}) 
# pull median price and count of top 5 neighborhoods.
grouped = data.groupby(['neighborhood']).agg({'price': "median", 'id':'count'})
grouped = grouped[grouped.id >= 10].groupby('neighborhood')['price'].median().sort_values(ascending=False)[:5] 
# pull top 5 neighborhoods by median price with at least 10 homes.
grouped
data.groupby(['superhost'])['superhost'].count() 
# pull the count of homes from superhost. 
# 't' for True being super host and 'f' for False not being superhost.