

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv('../input/train_2016_v2.csv')

prop_df = pd.read_csv('../input/properties_2016.csv')

samp = pd.read_csv('../input/sample_submission.csv')



print (train_df.head())

print (prop_df.head())  
print(train_df.columns)

print(prop_df.columns)

print(train_df.shape)

print(prop_df.shape)
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

print(train_df.head())

print(train_df.shape)
count = 0

for c in list(train_df):

    if (len(train_df[c].unique()) == 1):

        print(c)

        count+=1

print(count)
count = 0

low_var_cols = []

for c in list(train_df):

    if (len(train_df[c].unique()) < 907):

        print(c)

        low_var_cols.append(c)

        count+=1

print(count)
len(low_var_cols)
count = 0

low_var_drop_cols = []

for c in low_var_cols:

    if (train_df[c].nunique() <= 2):

        print(c)

        low_var_drop_cols.append(c)

        count+=1

print(count)

len(low_var_drop_cols)
print(train_df['assessmentyear'].nunique())
#--- List of columns having Nan values and the number ---



missing_col = train_df.columns[train_df.isnull().any()].tolist()

print(missing_col)

print('There are {} missing columns'.format(len(missing_col)))
nonmissing_col = train_df.columns[~(train_df.isnull().any())].tolist()

print(nonmissing_col)

print('There are {} non-missing columns'.format(len(nonmissing_col)))
#--- Data type of each column ---

print(train_df.dtypes)
import seaborn as sns

#sns.barplot( x = train_df.dtypes.unique(), y = train_df.dtypes.value_counts(), data = train_df)

sns.barplot( x = ['float', 'object', 'int'], y = train_df.dtypes.value_counts(), data = prop_df)
print(train_df.dtypes.value_counts())
#--- Checking if all the parcelids are unique in both the dataframes ---



print (prop_df['parcelid'].nunique())

print (prop_df.shape[0])



print (train_df['parcelid'].nunique())

print (train_df.shape[0]) 

print(train_df['hashottuborspa'].nunique())

print(train_df['hashottuborspa'].unique())

print('\n')

print(train_df['propertycountylandusecode'].nunique())

print(train_df['propertycountylandusecode'].unique())

print('\n')

print(train_df['propertyzoningdesc'].nunique())

print(train_df['propertyzoningdesc'].unique())

print('\n')

print(train_df['fireplaceflag'].nunique())

print(train_df['fireplaceflag'].unique())

print('\n')

print(train_df['taxdelinquencyflag'].nunique())

print(train_df['taxdelinquencyflag'].unique()) 

print('\n') 

print(train_df['transactiondate'].nunique())

train_df['hashottuborspa'] = train_df['hashottuborspa'].fillna(0)

train_df['fireplaceflag'] = train_df['fireplaceflag'].fillna(0)

train_df['taxdelinquencyflag'] = train_df['taxdelinquencyflag'].fillna(0)



#---  replace the string 'True' and 'Y' with value '1' ---



train_df.hashottuborspa = train_df.hashottuborspa.astype(np.int8)

train_df.fireplaceflag = train_df.fireplaceflag.astype(np.int8)

train_df['taxdelinquencyflag'].replace( 'Y', 1, inplace=True)

train_df.taxdelinquencyflag = train_df.taxdelinquencyflag.astype(np.int8)
train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])



#--- Creating two additional columns each for the month and day ---

train_df['transaction_month'] = train_df.transactiondate.dt.month.astype(np.int64)

train_df['transaction_day'] = train_df.transactiondate.dt.weekday.astype(np.int64)



#--- Dropping the 'transactiondate' column now ---

train_df = train_df.drop('transactiondate', 1)
#--- Counting number of occurrences of Nan values in remaining two columns ---

print(train_df['propertycountylandusecode'].isnull().sum())

print(train_df['propertyzoningdesc'].isnull().sum())
#--- Since there is only ONE missing value in this column we will replace it manually ---

train_df["propertycountylandusecode"].fillna('023A', inplace =True)

print(train_df['propertycountylandusecode'].isnull().sum())
train_df["propertyzoningdesc"].fillna('UNIQUE', inplace =True)

print(train_df['propertyzoningdesc'].isnull().sum())
#--- Statistics of the target variable ---



print(train_df['logerror'].describe())
import matplotlib.pyplot as plt

plt.scatter(train_df['logerror'], train_df.logerror.values)

plt.xlabel('No of observations', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
#--- putting all columns of 'float' type in a list ---

float_cols = list(train_df.select_dtypes(include=['float']).columns)

print('There are {} columns of type float having missing values'.format(len(float_cols)))

print('\n')

print(float_cols)
#--- putting columns of type 'float' having missing values in a list ---

float_nan_col = []

for column in float_cols:

    if (train_df[column].isnull().sum() > 0):

        float_nan_col.append(column)



print('There are {} columns of type float having missing values'.format(len(float_nan_col)))

print('\n')

print(float_nan_col)
cols = ['regionidcity', 'regionidneighborhood', 'regionidzip']

print(train_df['regionidcity'].isnull().sum())

print(train_df['regionidneighborhood'].isnull().sum())

print(train_df['regionidzip'].isnull().sum())



train_df["regionidcity"].fillna(lambda x: np.random(train_df[train_df["regionidcity"] != np.nan]), inplace =True)

train_df["regionidneighborhood"].fillna(lambda x: np.random(train_df[train_df["regionidneighborhood"] != np.nan]), inplace =True)

train_df["regionidzip"].fillna(lambda x : np.random(train_df["regionidzip"] != np.nan) , inplace =True)



#--- cross check whether nan values are present or not ---

print(train_df['regionidcity'].isnull().sum())

print(train_df['regionidneighborhood'].isnull().sum())

print(train_df['regionidzip'].isnull().sum())
#--- some analysis on the column values ---



print(train_df['unitcnt'].unique())

print(train_df['unitcnt'].value_counts())

sns.countplot(x = 'unitcnt', data = train_df)
#--- Replace the missing values with the maximum occurences ---

train_df['unitcnt'] = train_df['unitcnt'].fillna(train_df['unitcnt'].mode()[0])



#--- cross check for missing values ---

print(train_df['unitcnt'].isnull().sum())
print(train_df['censustractandblock'].corr(train_df['rawcensustractandblock']))
print(train_df['censustractandblock'].nunique())

print(train_df['rawcensustractandblock'].nunique())
'''  #--- to be continued ---

print(train_df['censustractandblock'].isnull().sum())

#print('\n')

#print(train_df['rawcensustractandblock'].nunique())



#train_df['censustractandblock'] = train_df['censustractandblock'].fillna()

pop = pd.DataFrame()

pop['censustractandblock'] = train_df['censustractandblock'] 

print(pop.shape[0])



a = 0

count = 0

for i in pop['censustractandblock']:

    if (np.isnan(i)):

        a = train_df.iloc[count]['rawcensustractandblock']

        #a.append(train_df['rawcensustractandblock'].iloc())

        for j in pop['censustractandblock']:

            if ((np.isfinite(j)) & ( )):

                

        count+=1

print(count)

#pop['censustractandblock'] = pop['censustractandblock'].fillna(pop['censustractandblock'] /

       # if )

print (a)    

''' 

print(train_df['yearbuilt'].sort_values().unique())
train_df['yearbuilt'] = train_df['yearbuilt'].fillna(2016)



#--- cross check for missing values ---

print(train_df['yearbuilt'].isnull().sum())
#--- list of columns of type 'float' having missing values

#--- float_nan_col 



#--- list of columns of type 'float' after imputing missing values ---

float_filled_cols = ['regionidcity', 'regionidneighborhood', 'regionidzip', 'unitcnt', 'censustractandblock', 'yearbuilt']



count = 0

for i in float_nan_col:

    if i not in float_filled_cols:

        train_df[i] = train_df[i].fillna(0)

        count+=1

print(count)
print(len(float_nan_col))
sns.regplot(x = 'latitude', y = 'longitude', data = train_df)
x = train_df.iloc[1]

#print(x)
#--- how old is the house? ---

train_df['house_age'] = 2017 - train_df['yearbuilt']



#--- how many rooms are there? ---  

train_df['tot_rooms'] = train_df['bathroomcnt'] + train_df['bedroomcnt']



#--- does the house have A/C? ---

train_df['AC'] = np.where(train_df['airconditioningtypeid']>0, 1, 0)



#--- Does the house have a deck? ---

train_df['deck'] = np.where(train_df['decktypeid']>0, 1, 0)

train_df.drop('decktypeid', axis=1, inplace=True)



#--- does the house have a heating system? ---

train_df['heating_system'] = np.where(train_df['heatingorsystemtypeid']>0, 1, 0)



#--- does the house have a garage? ---

train_df['garage'] = np.where(train_df['garagecarcnt']>0, 1, 0)



#--- does the house come with a patio? ---

train_df['patio'] = np.where(train_df['yardbuildingsqft17']>0, 1, 0)



#--- does the house have a pool?

train_df['pooltypeid10'] = train_df.pooltypeid10.astype(np.int8)

train_df['pooltypeid7'] = train_df.pooltypeid7.astype(np.int8)

train_df['pooltypei2'] = train_df.pooltypeid2.astype(np.int8)

train_df['pool'] = train_df['pooltypeid10'] | train_df['pooltypeid7'] | train_df['pooltypeid2'] 



#--- does the house have all of these? -> spa/hot-tub/pool, A/C, heating system , garage, patio

train_df['exquisite'] = train_df['pool'] + train_df['patio'] + train_df['garage'] + train_df['heating_system'] + train_df['AC'] 



#--- Features based on location ---

train_df['x_loc'] = np.cos(train_df['latitude']) * np.cos(train_df['longitude'])

train_df['y_loc'] = np.cos(train_df['latitude']) * np.sin(train_df['longitude'])

train_df['z_loc'] = np.sin(train_df['latitude'])



print('DONE')
#train_df['transaction_year']

sns.countplot(x = 'transaction_month', data = train_df)
#--- create an additional feature called season ---

def seas(x):

    if 2 < x < 6:

        return 1        #--- Spring

    elif 5 < x < 9:

        return 2        #---Summer

    elif 8 < x < 12:

        return 3        #--- Fall (Autumn) 

    else:

        return 4        #--- Winter 



train_df['season'] = train_df['transaction_month'].apply(seas)
ax = sns.countplot(x = 'season', data = train_df)

ax.set(xlabel='Seasons', ylabel='Count')

season_list=['Spring','Summer','Fall','Winter']

plt.xticks(range(4), season_list, rotation=45)

plt.show()
ax = sns.countplot(x = 'exquisite', data = train_df)

ax.set(xlabel='Exquisite features present', ylabel='Count')

plt.show()
ax = sns.countplot(x = 'transaction_day', data = train_df)

ax.set(xlabel='Transaction Days', ylabel='Count')

days_list=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.xticks(range(len(days_list)), days_list, rotation=45)

plt.show()
#--- create an additional feature called weekday_trans ---

def weekday_transaction(x):

    if 4 < x <= 6:

        return 1        #--- Weekend

    else:

        return 2        #--- Weekday



train_df['weekday_trans'] = train_df['transaction_day'].apply(weekday_transaction)
ax = sns.countplot(x = 'weekday_trans', data = train_df)

ax.set(xlabel='Weekday/weekend', ylabel='Count')

weekend_day_list=['Weekend', 'Weekday']

plt.xticks(range(len(weekend_day_list)), weekend_day_list, rotation=45)

plt.show()
#--- living area ---

train_df['LivingArea'] = train_df['calculatedfinishedsquarefeet']/train_df['lotsizesquarefeet']

train_df['LivingArea_2'] = train_df['finishedsquarefeet12']/train_df['finishedsquarefeet15']



#--- Extra space available

train_df['ExtraSpace'] = train_df['lotsizesquarefeet'] - train_df['calculatedfinishedsquarefeet'] 

train_df['ExtraSpace-2'] = train_df['finishedsquarefeet15'] - train_df['finishedsquarefeet12'] 
#Ratio of tax of property over parcel

train_df['ValueRatio'] = train_df['taxvaluedollarcnt']/train_df['taxamount']



#TotalTaxScore

train_df['TaxScore'] = train_df['taxvaluedollarcnt']*train_df['taxamount']
#Number of properties in the zip

zip_count = train_df['regionidzip'].value_counts().to_dict()

train_df['zip_count'] = train_df['regionidzip'].map(zip_count)



#Number of properties in the city

city_count = train_df['regionidcity'].value_counts().to_dict()

train_df['city_count'] = train_df['regionidcity'].map(city_count)



#Number of properties in the city

region_count = train_df['regionidcounty'].value_counts().to_dict()

train_df['county_count'] = train_df['regionidcounty'].map(region_count)
#--- Number of columns present in our dataframe now ---

a = train_df.columns.tolist()

print('Now there are {} columns in our dataframe'.format(len(a)))
import math

p = pd.DataFrame()

p['val'] = np.exp(train_df['logerror'])

print(p.head())



plt.scatter(p['val'], p.val.values)

plt.xlabel('No of observations', fontsize=12)

plt.ylabel('vals', fontsize=12)

plt.show()
print(p.describe())
p = p[p['val'] < 40]



plt.scatter(p['val'], p.val.values)

plt.xlabel('No of observations', fontsize=12)

plt.ylabel('vals', fontsize=12)

plt.show()



print(p.describe())
#plt.hist(np.log(train_df['trip_duration']+25), bins = 25)

plt.hist(train_df['logerror'], bins = 100)
corr = train_df.corr()

fig, ax = plt.subplots(figsize=(20, 20))

ax.matshow(corr)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

plt.xticks(range(len(corr.columns)), corr.columns, fontsize = 15)

plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 15)
alist = []

alist = train_df['yearbuilt'].unique()

alist.sort()
from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(20, 20))

ax = sns.countplot(x = 'yearbuilt', data = train_df)

ax.set(xlabel='Year Built', ylabel='Count')

#weekend_day_list=['Weekend', 'Weekday']

plt.xticks(range(len(alist)), alist, rotation=90)

plt.show()
'''

from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(20, 20))

ax = sns.countplot(x = 'yearbuilt', data = train_df)

ax.set(xlabel='Year Built', ylabel='Count')

#weekend_day_list=['Weekend', 'Weekday']

plt.xticks(range(len(alist)), alist, rotation=90)

plt.show()

''' 

'''

x = list(train_df['yearbuilt'])

y = train_df['logerror']

fig = plt.bar(x, y)

plt.show()

'''
#--- Memory usage of entire dataframe ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
#--- Memory usage of each column ---

print(train_df.memory_usage()/ 1024**2)  #--- in MB ---
#--- List of columns that cannot be reduced in terms of memory size ---

count = 0

for col in train_df.columns:

    if train_df[col].dtype == object:

        count+=1

        print (col)

print('There are {} columns that cannot be reduced'.format(count))        
count = 0

for col in train_df.columns:

    if train_df[col].dtype != object:

        if ((train_df[col].max() < 255) & (train_df[col].min() > -255)):

            if((col != 'logerror')|(col != 'yearbuilt')|(col != 'xloc')|(col != 'yloc')|(col != 'zloc')):

                count+=1

                train_df[col] = train_df[col].astype(np.int8)

                print (col)

print(count)                

                
#--- Memory usage of reduced dataframe ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
#--- Reducing memory of `float64` type columns to `float32` type columns



count = 0

for col in train_df.columns:

    if train_df[col].dtype != object:

        if train_df[col].dtype == float:

            train_df[col] = train_df[col].astype(np.float32)

            count+=1

print('There were {} such columns'.format(count))
#--- Let us check the memory consumed again ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
#print(train_df.dtypes)

#print(train_df.dtypes.value_counts())

col_int64 = []

for col in train_df.columns:

    if train_df[col].dtype == 'int64':

        print(col)

        col_int64.append(col)

print(col_int64)
for i in col_int64:

    print('{} - {} and {}'.format(i, max(train_df[i]), min(train_df[i])) )
train_df['zip_count'] = train_df['zip_count'].astype(np.int32)

train_df['city_count'] = train_df['city_count'].astype(np.int32)

train_df['county_count'] = train_df['county_count'].astype(np.int32)
#--- Let us check the memory consumed again ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")