import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
with open("../input/data.dat") as json_file:
    json_data = json.load(json_file)
print(type(json_data))
with open("../input/data.dat") as json_file:
    json_data = json.load(json_file)
json_data['houses'] 
df = json_normalize(json_data, 'houses') # parsing the data into dataframe format
df.head()
# get some general insights about the dataset
df.info()
# format date for values in column 'date'
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%dT%H%M%S", errors = 'coerce')
df.info()
# identify row indexes of missing values in column date 
df[(df.date.isnull() == True)]
# parsing the data again into table 
json_data['houses']
df = json_normalize(json_data, 'houses') # parsing the data into dataframe format

# display 2 row indexes that have wrong value format
df[4334:4336]
df.loc[4334, 'date'] = '20140630T000000'
df.loc[4335, 'date'] = '20140523T000000'
# format date for values in column 'date'
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%dT%H%M%S", errors = 'ignore')
# convert data type of column yr_renovated to integer
# if value is missing, set it to 0
df.yr_renovated = df.yr_renovated.apply(lambda x: int(x) if x == x else 0)

# convert data type of column price to integer
# if value is missing, set it to 0
df.price= df.price.apply(lambda x: int(x) if x == x else 0)

df.head()
import re

# extract number of bedrooms for each address/house and store in a list
bed_rooms = []
for bed in df['rooms']:
    num = re.search(r'bedrooms:\s+(\d*\.\d+|\d+)', bed)
    bed_rooms.append(float(num.group(1)))

# extract number of bathrooms for each address/house and store in a list
bath_rooms = []
for bath in df['rooms']:
    num2 = re.search(r'bathrooms:\s+(\d*\.\d+|\d+)', bath)
    bath_rooms.append(float(num2.group(1)))
# add 2 new columns to the dataframe
df['bathrooms'] = bath_rooms
df['bedrooms'] = bed_rooms

# df.head()
# Compare newly added columns with the original one. Numbers are put in correct columns for bathrooms and bedrooms number. 
df = df.drop(['rooms'], 1)
df.head()
# split column 'area' into 3 different columns: 'sqft_basement', 'sqft_above' and 'sqft_living/sqft_lot'
# then drop the 'area' column
df[['sqft_basement','sqft_above','sqft_living/sqft_lot']] = pd.DataFrame(df['area'].values.tolist(), columns=['sqft_basement','sqft_above','sqft_living/sqft_lot'])
df = df.drop('area', axis = 1)

# continue to split column 'sqft_living/sqft_lot' and drop it
df[['living/lot', 'sqft_num']] = df['sqft_living/sqft_lot'].str.split('=', expand=True)
df = df.drop(['sqft_living/sqft_lot','living/lot'], 1)

# finally, split 'sqft_num' column and drop it
df[['sqft_living', 'sqft_lot']] = df['sqft_num'].str.split("\\", expand=True)
df[['sqft_living', 'sqft_lot']] = df[['sqft_living', 'sqft_lot']].apply(pd.to_numeric)
df = df.drop('sqft_num', 1)

df.head()
# split column 'address' into 4 columns
df[['street','city','statezip','country']] = df['address'].str.split(', ',expand=True)
df = df.drop('address', axis=1)

# reorder columns in the dataframe
# then, check value type of each column for further processing
df = df[["date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", 
         "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "street", "city", "statezip", "country"]]
print(len(df.columns))
df.dtypes
# checking number of unique values in column 'country'
print(len(df['country'].unique()))
print(df['country'].unique())
df = df.drop('country', 1)
df.head()
df['city'].value_counts()
# sorting the list of unique city names in the dataframe 
print(sorted(df['city'].unique().tolist()))
# Fix name errors in column 'city'
df['city'].replace({'Auburnt':'Auburn', 'auburn':'Auburn', 
                 'Belleview':'Bellevue', 'Bellvue':'Bellevue',
                 'Issaguah':'Issaquah',
                 'Kirklund':'Kirkland',
                 'Redmonde':'Redmond', 'Redmund':'Redmond', 'redmond':'Redmond',
                 'Samamish':'Sammamish', 'sammamish':'Sammamish',
                 'Seaattle':'Seattle', 'Seatle':'Seattle', 'seattle':'Seattle',
                 'Snogualmie':'Snoqualmie', 'Snoqualmie Pass':'Snoqualmie',
                 'Woodenville':'Woodinville'}, inplace=True)
print(sorted(df['city'].unique().tolist()))
len(df['statezip'].unique())
df[(df['sqft_above'] + df['sqft_basement'] != df['sqft_living'])]
# replacing values for 2 rows at index 4338 and 4339
df.loc[4338, 'sqft_living'] = 2700
df.loc[4339, 'sqft_basement'] = 300
df.loc[4337:4340]
df[(df['yr_renovated'] != 0) & (df['yr_renovated'] <= df['yr_built'])]
# swap values of 2 columns yr_built and yr_renovated
df.loc[4340, 'yr_built'], df.loc[4340, 'yr_renovated'] = df.loc[4340, 'yr_renovated'], df.loc[4340, 'yr_built']
df.loc[4341, 'yr_built'], df.loc[4341, 'yr_renovated'] = df.loc[4341, 'yr_renovated'], df.loc[4341, 'yr_built']
df.loc[4342, 'yr_built'], df.loc[4342, 'yr_renovated'] = df.loc[4342, 'yr_renovated'], df.loc[4342, 'yr_built']
df.loc[4345, 'yr_built'], df.loc[4345, 'yr_renovated'] = df.loc[4345, 'yr_renovated'], df.loc[4345, 'yr_built']

df.loc[4340:4345]
df[df.duplicated(["price", "street"], keep=False)]
# display columns that will be dropped
df[df.duplicated(["price", "street"], keep="last")]
# drop the 2 data rows above
df.drop_duplicates(["price", "street"], keep='last', inplace=True)
df[df.duplicated(["date", "street"], keep=False)]
df.drop_duplicates(["date", "street"], keep='first', inplace=True)
# check the length of dataframe after dropping duplicated rows
len(df)
df[(df['bedrooms']==0) | (df['bathrooms']==0)]
%matplotlib inline
df['bedrooms'].hist()
df['bathrooms'].hist()
# check the information of properties that have price in range from $1,000,000 to $1,500,000
new_df = df[(df['price']>=1000000) & (df['price']<=1500000)]
new_df
# display properties with statezip of WA 98102
new_df[(new_df['statezip'] == 'WA 98102')]
# replace missing values
df.loc[2365, 'bedrooms'] = 3
df.loc[2365, 'bathrooms'] = 3

# display properties with statezip of WA 98053
new_df[(new_df['statezip'] == 'WA 98053')]
# replace missing values
df.loc[3209, 'bedrooms'] = 3
df.loc[3209, 'bathrooms'] = 3.53

# check if missing values of bedrooms and bathrooms still exist in the dataframe
df[(df['bedrooms']==0) | (df['bathrooms']==0)]
# display histogram of price
plt.figure(figsize=(15, 10))
df.price.hist(bins=500)
plt.xlim(0,2000000)
len(df[(df.price == 0)])
# replace value 0 in column price by NaN 
df.loc[4353:4600,'price'] = df.loc[4353:4600,'price'].replace(0, np.NaN)
df.loc[4353:4600]
# create a copy of the initial dataframe
impute_df = df.copy()

# drop null values from initial dataframe
df.dropna(subset=['price'],axis=0,inplace=True)
df['price'].isnull().sum()

# null values still exist in imputed dataframe
impute_df['price'].isnull().sum()

# drop column indexes that are inappropriate in the model as they are not in numerical format 
lm_fitting_df = df.drop(['date','street','city','statezip'],axis=1)

lm_for_impute = LinearRegression() #instatiate
lm_for_impute.fit(lm_fitting_df[[x for x in lm_fitting_df.columns if x != 'price']],lm_fitting_df['price']) #fit

impute_df[impute_df['price'].isnull()].head()
lm_for_impute.predict(impute_df.drop(['price','statezip','city','street','date'],axis=1)) 
#this uses the other features to predict 'price' with the model

impute_df['price'][impute_df['price'].isnull()] = lm_for_impute.predict(impute_df.drop(['price','statezip','city','street','date'],axis=1))

# display boxplot to compare the dropna dataframe and imputed dataframe
boxplot = pd.DataFrame({'imputed': impute_df['price'], 'dropped': df['price']})
boxplot.plot(kind='box')
# set the y limit a bit smaller
boxplot = pd.DataFrame({'imputed': impute_df['price'], 'dropped': df['price']})
boxplot.plot(kind='box')
plt.ylim(0,2000000)
# convert data type of column price to integer
# if value is not available, set it to 0
impute_df.price = impute_df.price.apply(lambda x: int(x) if x == x else 0)

# display rows that previously had missing value in original dataframe
impute_df.loc[4353:4601]
# display boxplot for yr_built and yr_renovated
plt.figure(figsize=(10, 7))
graph = impute_df[['yr_built','yr_renovated']].boxplot()
plt.ylim(1890,2020)
# boxplot number of bedrooms, bathrooms and floors
plt.figure(figsize=(10, 7))
graph = impute_df[['bedrooms','bathrooms','floors']].boxplot()
# display properties which have number of either bedrooms or bathrooms equal or more than 8
impute_df[(impute_df['bedrooms'] >= 8) | (impute_df['bathrooms'] >= 8)]
# boxplot price
plt.figure(figsize=(10, 7))
graph = impute_df[['price']].boxplot()
# display sales record that have price more than 5 mil dollar
impute_df[(impute_df['price'] >= 5000000)]
impute_df['price'].max()
# drop 2 rows that have suspicious values
impute_df.drop(4347, axis = 0, inplace=True)
impute_df.drop(4351, axis = 0, inplace=True)
impute_df = impute_df.reset_index(drop=True)

# display boxplot graph for values of square footage 
plt.figure(figsize=(10, 7))
graph = impute_df[['sqft_living','sqft_lot','sqft_above','sqft_basement']].boxplot()
# display rows which have sqft_lot larger than 600,000
impute_df[(impute_df['sqft_lot'] >= 600000)]
impute_df.drop(2478, axis = 0, inplace=True)

# reset index and writing cleaned data to csv format file
impute_df.reset_index()
impute_df.to_csv("RealEstateData.csv", encoding='utf-8')