import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from sklearn.model_selection import train_test_split

import gc



import lightgbm as lgb

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

color = sns.color_palette()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train2016_df = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv", parse_dates=["transactiondate"])

train2017_df = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv", parse_dates=["transactiondate"])

print ('train 2016 data has {0} rows and {1} columns'.format(train2016_df.shape[0],train2016_df.shape[1]))

print ('----------------------------')

print ('train 2017 data has {0} rows and {1} columns'.format(train2017_df.shape[0],train2017_df.shape[1]))
train2016_df.head()
(train2016_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()
# Look at the distribution of the target variable (log-error)

print(train2016_df['logerror'].describe())
print('Skewness is', train2016_df['logerror'].skew())

target = train2016_df.loc[abs(train2016_df['logerror']) < 0.4, 'logerror']

print('Skewness after tranforms is', target.skew())

print('train data has rows', target.shape)

target.hist(bins=40)
# biểu diễn tháng sales bằng bar plot

train2016_df['transaction_month'] = train2016_df['transactiondate'].dt.month

cnt_srs = train2016_df['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()

train2016_df.drop(['transaction_month'], axis=1, inplace=True)
# check the distribution of the target variable logerror 2017

print('Skewness is', train2017_df['logerror'].skew())

target = train2017_df.loc[abs(train2017_df['logerror']) < 0.4, 'logerror']

print('Skewness after tranforms is', target.skew())

print('train data has rows', target.shape)

sns.distplot(target)
prop2016_df = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")

prop2016_df.shape
pd.set_option('display.max_columns', None)

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

prop2016_df.head()
prop2017_df = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")

prop2017_df.shape
# merge 2 file

train_2016 = train2016_df.merge(prop2016_df, how='left', on='parcelid')

train_2017 = train2017_df.merge(prop2017_df, how='left', on='parcelid')
dicts = pd.read_excel('/kaggle/input/zillow-prize-1/zillow_data_dictionary.xlsx')

dicts.head(10)
catvars = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid',

           'decktypeid','fips','hashottuborspa', 'fireplaceflag','heatingorsystemtypeid',

           'propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','regionidcity',

           'regionidcounty','regionidneighborhood','regionidzip','storytypeid','typeconstructiontypeid','yearbuilt',

           'taxdelinquencyflag', 'latitude', 'longitude', 'parcelid', 'assessmentyear']



numvars = [i for i in prop2016_df.columns if i not in catvars]

print ("Có {} numeric và {} categorical columns".format(len(numvars),len(catvars)))
corr = prop2016_df[numvars].corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(19, 19))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

cmap ='coolwarm'



# Draw the heatmap with the mask and correct aspect ratio

ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=True,

            square=True, linewidths=.3, cbar_kws={"shrink": .5})

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
del corr, target, ax, dicts

gc.collect()

print('Memory usage reduction…')
# create numeric plots

sns.set(style="whitegrid", color_codes=True)

nd = pd.melt(train_2016, value_vars = numvars)

n1 = sns.FacetGrid(nd, col='variable', col_wrap=6, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
del nd, n1

gc.collect()

print('Memory usage reduction…')
# Missing value trong từng cột

missing_df = prop2016_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
# miss > 99%

missing_df = prop2016_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = missing_df['missing_count'] / prop2016_df.shape[0]

missing_df.loc[missing_df['missing_ratio']>0.99]
def poolhottubor_process(property_data):

    # 0 pools

    property_data.poolcnt.fillna(0,inplace = True)

    # 0 hot tubs or spas

    property_data.hashottuborspa.fillna(0,inplace = True)

    # Convert "True" to 1

    property_data.hashottuborspa.replace(to_replace = True, value = 1,inplace = True)



    # Set properties that have a pool but no info on poolsize equal to the median poolsize value.

    property_data.loc[property_data.poolcnt==1, 'poolsizesum'] = property_data.loc[property_data.poolcnt==1, 'poolsizesum'].fillna(property_data[property_data.poolcnt==1].poolsizesum.median())

    # "0 pools" = "0 sq ft of pools"

    property_data.loc[property_data.poolcnt==0, 'poolsizesum']=0



    # "0 pools with a spa/hot tub"

    property_data.pooltypeid2.fillna(0,inplace = True)

    # "0 pools without a hot tub"

    property_data.pooltypeid7.fillna(0,inplace = True)



    # Drop redundant feature

    property_data.drop('pooltypeid10', axis=1, inplace=True)

    

    return property_data
# số lò sưởi trung bình 

print(prop2016_df['fireplacecnt'].value_counts())
def fireplace_process(property_data):

    # fireplaceflag là True và fireplacecnt NaN, ta sẽ thay thế bằng trung bình các fireplace.

    property_data.loc[(property_data['fireplaceflag'] == True) & (property_data['fireplacecnt'].isnull()), ['fireplacecnt']] = 1

    # If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".

    property_data.loc[(property_data['fireplacecnt'] >= 1.0) & (property_data['fireplaceflag'].isnull()), ['fireplaceflag']] = True

    

    # Convert "NaN" thành 0

    property_data.fireplaceflag.fillna(0,inplace = True)

    # Convert "True" thành 1

    property_data.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)

    

    # If 'fireplacecnt' is "NaN", replace with "0"

    property_data.fireplacecnt.fillna(0,inplace = True)

    

    return property_data
def garage_process(property_data):

    property_data.garagecarcnt.fillna(0,inplace = True)

    property_data.garagetotalsqft.fillna(0,inplace = True)

    return property_data
def tax_process(property_data):

    # Replace "NaN" with "0"

    property_data.taxdelinquencyflag.fillna(0,inplace = True)

    # Change "Y" to "1"

    property_data.taxdelinquencyflag.replace(to_replace = 'Y', value = 1,inplace = True)

    

    

    property_data.landtaxvaluedollarcnt.fillna(0,inplace = True)

    property_data.structuretaxvaluedollarcnt.fillna(0,inplace = True)



    property_data['taxvaluedollarcnt'].fillna((property_data['taxvaluedollarcnt'].mean()), inplace=True)

    

    # Drop "regionidcity"

    property_data.drop('regionidcity', axis=1, inplace=True)

    # Fill in "NaN" "yearbuilt" with most common

    yearbuilt = property_data['yearbuilt'].value_counts().idxmax()

    property_data['yearbuilt'] = property_data['yearbuilt'].fillna(yearbuilt)

    

    return property_data
squarefeet = prop2016_df[ prop2016_df['finishedsquarefeet15'].notnull() & prop2016_df['finishedsquarefeet50'].notnull() & prop2016_df['lotsizesquarefeet'].notnull()]

squarefeet[['calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet15','finishedsquarefeet50','numberofstories','lotsizesquarefeet']].sample(10)
def squarefeet_process(property_data):

    

    # Drop "finishedsquarefeet6"

    property_data.drop('finishedsquarefeet6', axis=1, inplace=True)

    # Drop "finishedsquarefeet12"

    property_data.drop('finishedsquarefeet12', axis=1, inplace=True)

    # Drop "finishedfloor1squarefeet"

    property_data.drop('finishedfloor1squarefeet', axis=1, inplace=True)



    # Replace "NaN" "calculatedfinishedsquarefeet" values with mean.

    property_data['calculatedfinishedsquarefeet'].fillna((property_data['calculatedfinishedsquarefeet'].mean()), inplace=True)



    # If "numberofstories" is equal to "1", then we can replace the "NaN"s with the "calculatedfinishedsquarefeet" value. Fill in the rest with the average values.

    property_data.loc[property_data['numberofstories'] == 1.0,'finishedsquarefeet50'] = property_data['calculatedfinishedsquarefeet']

    property_data['finishedsquarefeet50'].fillna((property_data['finishedsquarefeet50'].mean()), inplace=True)



    # Replace "NaN" "finishedsquarefeet15" values with calculatedfinishedsquarefeet.

    property_data.loc[property_data['finishedsquarefeet15'].isnull(),'finishedsquarefeet15'] = property_data['calculatedfinishedsquarefeet']

    # Replace rest valule "NaN" "finishedsquarefeet15" values with mean.

    property_data['finishedsquarefeet15'].fillna((property_data['finishedsquarefeet15'].mean()), inplace=True)

    # change numberofstories with common value 

    property_data.numberofstories.fillna(1,inplace = True)

    

    return property_data
plt.figure(figsize=(12,4))

sns.countplot(x="calculatedbathnbr", data=prop2016_df)

plt.ylabel('Count', fontsize=8)

plt.xlabel('Bathroom', fontsize=12)

plt.title('Frequency of bathroom count', fontsize=15)

plt.show()



# look at some data example 

bathrooms = prop2016_df[prop2016_df['fullbathcnt'].notnull() & prop2016_df['threequarterbathnbr'].notnull() & prop2016_df['calculatedbathnbr'].notnull()]

bathrooms[['fullbathcnt','threequarterbathnbr','calculatedbathnbr']].sample(10)
def bathroom_process(property_data):

    # Drop "threequarterbathnbr"

    property_data.drop('threequarterbathnbr', axis=1, inplace=True)

    # Drop "fullbathcnt"

    property_data.drop('fullbathcnt', axis=1, inplace=True)



    # Fill in "NaN" "calculatedbathnbr" with most common

    bathroommode = property_data['calculatedbathnbr'].value_counts().idxmax()

    property_data['calculatedbathnbr'] = property_data['calculatedbathnbr'].fillna(bathroommode)

    return property_data
# # identify levels of missingness

# missing = prop2016_df.isnull().sum().sort_values(ascending = False)

# vartypes = prop2016_df.dtypes

# missingpercent = (prop2016_df.isnull().sum()/prop2016_df.shape[0]).sort_values(ascending=False)

# pd.concat([vartypes, missing, missingpercent], axis = 1,

#           keys =['var type', 'missing n', 'percent']

#         ).sort_values(by = 'missing n', ascending = False).head(10)
def rest_process(property_data):

    # Drop "taxdelinquencyyear"

    property_data.drop('taxdelinquencyyear', axis=1, inplace=True)

    # Drop 'basementsqft'

    property_data.drop('basementsqft', axis=1, inplace = True)

    # Drop "storytypeid"

    property_data.drop('storytypeid', axis=1, inplace=True)

    # Drop "architecturalstyletypeid"

    property_data.drop('architecturalstyletypeid', axis=1, inplace=True)

    # Drop "typeconstructiontypeid" and "finishedsquarefeet13"

    property_data.drop('typeconstructiontypeid', axis=1, inplace=True)

    property_data.drop('finishedsquarefeet13', axis=1, inplace=True)

    # Drop "buildingclasstypeid"

    property_data.drop('buildingclasstypeid', axis=1, inplace=True)

    ##------------------------------------------------------------

    # Replace 'yardbuildingsqft17' "NaN"s with "0".

    property_data.yardbuildingsqft17.fillna(0,inplace = True)

    # Replace 'yardbuildingsqft26' "NaN"s with "0".

    property_data.yardbuildingsqft26.fillna(0,inplace = True)

    # Change "decktypeid" "Nan"s to "0"

    property_data.decktypeid.fillna(0,inplace = True)

    # Convert "decktypeid" "66.0" to "1"

    property_data.decktypeid.replace(to_replace = 66.0, value = 1,inplace = True)

    # change "airconditioningtypeid" NaN to "5"

    property_data.airconditioningtypeid.fillna(5,inplace = True)

    # change "heatingorsystemtypeid" NaN to "13"

    property_data.heatingorsystemtypeid.fillna(13,inplace = True)



    # Fill in "NaN" "buildingqualitytypeid" bằng giá trị phổ biến

    buildingqual = property_data['buildingqualitytypeid'].value_counts().idxmax()

    property_data['buildingqualitytypeid'] = property_data['buildingqualitytypeid'].fillna(buildingqual)

    # Fill in "NaN" "unitcnt" bằng giá trị phổ biến

    unitcommon = property_data['unitcnt'].value_counts().idxmax()

    property_data['unitcnt'] = property_data['unitcnt'].fillna(unitcommon)

    



    property_data['lotsizesquarefeet'].fillna((property_data['lotsizesquarefeet'].mean()), inplace=True)



    # Drop "regionidneighborhood"

    property_data.drop('regionidneighborhood', axis=1, inplace=True)

    # Drop 'regionidcounty'

    property_data.drop('regionidcounty', axis=1, inplace=True)

    

    return property_data
def fillcommonvalue(property_data):    

    # Drop "censustractandblock"

    property_data.drop('censustractandblock', axis=1, inplace=True)

    ##-------------------------------------------------------------

    # Fill in "regionidzip" bằng giá trị phổ biến

    regionidzip = property_data['regionidzip'].value_counts().idxmax()

    property_data['regionidzip'] = property_data['regionidzip'].fillna(regionidzip)



    # Fill in "fips" bằng giá trị phổ biến

    fips = property_data['fips'].value_counts().idxmax()

    property_data['fips'] = property_data['fips'].fillna(fips)



    # Fill in "propertylandusetypeid" bằng giá trị phổ biến

    propertylandusetypeid = property_data['propertylandusetypeid'].value_counts().idxmax()

    property_data['propertylandusetypeid'] = property_data['propertylandusetypeid'].fillna(propertylandusetypeid)



    # Fill in "latitude"  bằng giá trị phổ biến

    latitude = property_data['latitude'].value_counts().idxmax()

    property_data['latitude'] = property_data['latitude'].fillna(latitude)



    # Fill in "longitude" bằng giá trị phổ biến

    longitude = property_data['longitude'].value_counts().idxmax()

    property_data['longitude'] = property_data['longitude'].fillna(longitude)

    

    # Normal value

    property_data[['latitude', 'longitude']] /= 1e6

    property_data['rawcensustractandblock'] /= 1e6



    # Fill in "rawcensustractandblock" bằng giá trị phổ biến

    rawcensustractandblock = property_data['rawcensustractandblock'].value_counts().idxmax()

    property_data['rawcensustractandblock'] = property_data['rawcensustractandblock'].fillna(rawcensustractandblock)



    # Fill in "assessmentyear" bằng giá trị phổ biến

    assessmentyear = property_data['assessmentyear'].value_counts().idxmax()

    property_data['assessmentyear'] = property_data['assessmentyear'].fillna(assessmentyear)



    # Fill in "bedroomcnt" bằng giá trị phổ biến

    bedroomcnt = property_data['bedroomcnt'].value_counts().idxmax()

    property_data['bedroomcnt'] = property_data['bedroomcnt'].fillna(bedroomcnt)



    # Fill in "bathroomcnt" bằng giá trị phổ biến

    bathroomcnt = property_data['bathroomcnt'].value_counts().idxmax()

    property_data['bathroomcnt'] = property_data['bathroomcnt'].fillna(bathroomcnt)



    # Fill in "roomcnt" bằng giá trị phổ biến

    roomcnt = property_data['roomcnt'].value_counts().idxmax()

    property_data['roomcnt'] = property_data['roomcnt'].fillna(roomcnt)

    

    # Fill in "propertycountylandusecode" bằng giá trị phổ biến

    propertycountylandusecode = property_data['propertycountylandusecode'].value_counts().idxmax()

    property_data['propertycountylandusecode'] = property_data['propertycountylandusecode'].fillna(propertycountylandusecode)

    

    # Fill in "NaN" "propertyzoningdesc" with most common

    propertyzoningdesc = property_data['propertyzoningdesc'].value_counts().idxmax()

    property_data['propertyzoningdesc'] = property_data['propertyzoningdesc'].fillna(propertyzoningdesc)

    

    return property_data
# reduce 58 to 42 columns 

prop2016_df.shape
def convert_transactiondate(train_with_months):

    train_with_months['sale_month'] = train_with_months['transactiondate'].apply(lambda x: (x.to_pydatetime()).month)

    train_with_months['sale_day'] = train_with_months['transactiondate'].apply(lambda x: (x.to_pydatetime()).day)

    train_with_months['sale_year'] = train_with_months['transactiondate'].apply(lambda x: (x.to_pydatetime()).year)

    train_with_months.drop(['transactiondate'],axis=1,inplace=True)

    return train_with_months
def preprocess_data(property_data):

    property_data = poolhottubor_process(property_data)

    property_data = fireplace_process(property_data)

    property_data = garage_process(property_data)

    property_data = tax_process(property_data)

    property_data = squarefeet_process(property_data)

    property_data = bathroom_process(property_data)

    

    property_data = rest_process(property_data)

    property_data = fillcommonvalue(property_data)

    return property_data
prop2016 = preprocess_data(prop2016_df)

prop2017 = preprocess_data(prop2017_df)



print ('prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))
# identify levels of missingness

missing = prop2016.isnull().sum().sort_values(ascending = False)

vartypes = prop2016.dtypes

missingpercent = (prop2016.isnull().sum()/prop2016.shape[0]).sort_values(ascending=False)

pd.concat([vartypes, missing, missingpercent], axis = 1,

          keys =['var type', 'missing n', 'percent']

        ).sort_values(by = 'missing n', ascending = False).head(10)
for c in prop2016.columns:

    if prop2016[c].dtype == 'object':

        print(c)
from sklearn.preprocessing import LabelEncoder

countylandusecode = LabelEncoder()

prop2016["propertycountylandusecode"] = countylandusecode.fit_transform(prop2016["propertycountylandusecode"])

prop2017["propertycountylandusecode"] = countylandusecode.fit_transform(prop2017["propertycountylandusecode"])



zoningdesc = LabelEncoder()

prop2016["propertyzoningdesc"] = zoningdesc.fit_transform(prop2016["propertyzoningdesc"])

prop2017["propertyzoningdesc"] = zoningdesc.fit_transform(prop2017["propertyzoningdesc"])

print(prop2016_df['propertyzoningdesc'].value_counts())
prop2016_df['parcelid']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

prop2016 = scaler.fit_transform(prop2016.loc[:, 'airconditioningtypeid':])

prop2016 = pd.DataFrame(prop2016, columns=prop2016_df.loc[:, 'airconditioningtypeid':].columns)

prop2016['parcelid'] = prop2016_df['parcelid']

prop2017 = scaler.fit_transform(prop2017.loc[:, 'airconditioningtypeid':])

prop2017 = pd.DataFrame(prop2017, columns=prop2017_df.loc[:, 'airconditioningtypeid':].columns)

prop2017['parcelid'] = prop2017_df['parcelid']
prop2016.head()
def feature_engineering(property_data):

    property_data['avg_garage_size'] = property_data['garagetotalsqft'] / property_data['garagecarcnt']

    property_data['avg_garage_size'].fillna(0, inplace=True)

    

    # Rotated Coordinates

    property_data['location_1'] = property_data['latitude'] + property_data['longitude']

    property_data['location_2'] = property_data['latitude'] - property_data['longitude']

    property_data['location_3'] = property_data['latitude'] + 0.5 * property_data['longitude']

    property_data['location_4'] = property_data['latitude'] - 0.5 * property_data['longitude']



    property_data['taxpercentage'] = property_data['taxamount'] / property_data['taxvaluedollarcnt']

    property_data['taxpercentage'].fillna((property_data['taxpercentage'].mean()), inplace=True)

    # Drop "taxamount"

    property_data.drop('taxamount', axis=1, inplace=True)



    # Thêm derived room_cnt feature bằng cách cộng bathroom_cnt và bedroom_cnt

    property_data['derived_room_cnt'] = property_data['bedroomcnt'] + property_data['bathroomcnt']

    

    return property_data



prop2016 = feature_engineering(prop2016)

prop2017 = feature_engineering(prop2017)

print ('prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))

print ('prop 2017 data has {0} rows and {1} columns'.format(prop2017.shape[0],prop2017.shape[1]))
# import featuretools as ft

# # creating and entity set 'es'

# es = ft.EntitySet(id = 'zillow')



# # adding a dataframe 

# es.entity_from_dataframe(entity_id = 'properties', dataframe = prop2016, index = 'parcelid')

# # create entity

#es.normalize_entity(base_entity_id = 'properties', new_entity_id='land_zone', index = 'propertyzoningdesc', additional_variables = ['propertycountylandusecode', 'propertylandusetypeid'])
# feature_matrix, feature_names = ft.dfs(entityset=es, 

#                                       target_entity = 'properties', 

#                                       max_depth = 2, 

#                                       verbose = 1)
train_2016 = convert_transactiondate(train2016_df.copy()).merge(prop2016, how='left', on='parcelid')

train_2017 = convert_transactiondate(train2017_df.copy()).merge(prop2017, how='left', on='parcelid')

train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)



print ('prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))

print ('train 2016 data has {0} rows and {1} columns'.format(train_2016.shape[0],train_2016.shape[1]))
dtype_df = prop2016.dtypes.reset_index()

dtype_df.columns = ['Count', 'Column type']

dtype_df.groupby('Column type').aggregate('count').reset_index()
# export prop after data processing

prop2016.to_csv("properties_2016_proc.csv.gz", index=False, compression='gzip')

prop2017.to_csv("properties_2017_proc.csv.gz", index=False, compression='gzip')
for c in train.columns:

    if train[c].dtype != 'object' and c in catvars:

        print("{0} have type: {1}".format(c,train[c].dtype ))
# to change use .astype() 

# CatBoost requires all the categorical variables to be in the string format

def float2string(train):

    train['airconditioningtypeid'] = train.airconditioningtypeid.astype(str)

    train['buildingqualitytypeid'] = train.buildingqualitytypeid.astype(str)

    train['decktypeid'] = train.decktypeid.astype(str)

    train['fips'] = train.fips.astype(str)

    train['hashottuborspa'] = train.fips.astype(str)

    train['heatingorsystemtypeid'] = train.heatingorsystemtypeid.astype(str)

    train['latitude'] = train.latitude.astype(str)

    train['longitude'] = train.longitude.astype(str)

    train['propertycountylandusecode'] = train.fips.astype(str)

    train['propertylandusetypeid'] = train.fips.astype(str)

    train['propertyzoningdesc'] = train.fips.astype(str)

    train['regionidzip'] = train.regionidzip.astype(str)

    train['yearbuilt'] = train.yearbuilt.astype(str)

    train['fireplaceflag'] = train.fips.astype(str)

    train['assessmentyear'] = train.assessmentyear.astype(str)

    train['taxdelinquencyflag'] = train.fips.astype(str)

    

    return train

train_cat = float2string(train)
prop2016 = float2string(prop2016)

prop2017 = float2string(prop2017)
catboot_features = train_cat.drop(['parcelid', 'logerror', 'sale_month', 'sale_day', 'sale_year'], axis=1)



print("Number of features for CastBoot: {}".format(len(catboot_features.columns)))
# Prepare training and cross-validation data

catboot_label = train_cat.logerror.astype(np.float32)

print(catboot_label.head())



# Transform to Numpy matrices

catboot_X = catboot_features.values

catboot_y = catboot_label.values



# Perform shuffled train/test split

np.random.seed(42)

random.seed(10)

X_train, X_val, y_train, y_val = train_test_split(catboot_X, catboot_y, test_size=0.2)



# Remove outlier examples from X_train and y_train; Keep them in X_val and y_val for proper cross-validation

outlier_threshold = 0.4

mask = (abs(y_train) <= outlier_threshold)

X_train = X_train[mask, :]

y_train = y_train[mask]



print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

print("X_val shape: {}".format(X_val.shape))

print("y_val shape: {}".format(y_val.shape))
del prop2016_df, prop2017_df, catboot_X, catboot_y

gc.collect()

print('Memory usage reduction...')
feature_names = [s for s in catboot_features.columns]



categorical_indices = []

for i, n in enumerate(catboot_features.columns):

    if n in catvars:

        categorical_indices.append(i)

print(categorical_indices)
# CatBoost parameters after tuned hyperparameter

params = {}

params['loss_function'] = 'MAE'

params['eval_metric'] = 'MAE'

params['nan_mode'] = 'Min'  # Method to handle NaN (set NaN to either Min or Max)

params['random_seed'] = 0



params['iterations'] = 1000  # default 1000, use early stopping during training

params['learning_rate'] = 0.03  # default 0.03

params['max_depth'] = 10  # default 6 (must be <= 16, 6 to 10 is recommended)

params['l2_leaf_reg'] = 9  # default 3 (used for leaf value calculation, try different values)



params['border_count'] = 254  # default 254 (alias max_bin, suggested to keep at default for best quality)

params['bagging_temperature'] = 1  # default 1 (higher value -> more aggressive bagging, try different values)
%%time

from catboost import CatBoostRegressor, Pool

# Train CatBoost Regressor with cross-validated early-stopping

val_pool = Pool(X_val, y_val, cat_features=categorical_indices)



np.random.seed(42)

random.seed(36)

model = CatBoostRegressor(**params)

model.fit(X_train, y_train,

          cat_features=categorical_indices,

          use_best_model=True, eval_set=val_pool, early_stopping_rounds=30,

          logging_level='Silent', plot=True)



# Evaluate model performance

print("Train score: {}".format(abs(model.predict(X_train) - y_train).mean() * 100))

print("Val score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))
def predict_and_export(models, features_2016, features_2017, file_name):

    # Construct DataFrame for prediction results

    submission_2016 = pd.DataFrame()

    submission_2017 = pd.DataFrame()

    submission_2016['ParcelId'] = features_2016.parcelid

    submission_2017['ParcelId'] = features_2017.parcelid

    

    test_features_2016 = features_2016.drop(['parcelid'], axis=1)

    test_features_2017 = features_2017.drop(['parcelid'], axis=1)

    

    pred_2016, pred_2017 = [], []

    for i, model in enumerate(models):

        print("Start model {} (2016)".format(i))

        pred_2016.append(model.predict(test_features_2016))

        print("Start model {} (2017)".format(i))

        pred_2017.append(model.predict(test_features_2017))

    

    # Take average across all models

    mean_pred_2016 = np.mean(pred_2016, axis=0)

    mean_pred_2017 = np.mean(pred_2017, axis=0)

    

    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]

    submission_2016['201611'] = submission_2016['201610']

    submission_2016['201612'] = submission_2016['201610']



    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]

    submission_2017['201711'] = submission_2017['201710']

    submission_2017['201712'] = submission_2017['201710']

    

    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')

    

    print("Length of submission DataFrame: {}".format(len(submission)))

    print("Submission header:")

    print(submission.head())

    

    submission.to_csv(file_name, index=False, compression='gzip')

    return submission, pred_2016, pred_2017
train_cat.sample(10)
prop2016.sample(10)
%%time

file_name = 'v23_EDA_catboost_single.csv.gz'

submission, pred_2016, pred_2017 = predict_and_export([model], prop2016, prop2017, file_name)
del model, submission, pred_2016, pred_2017

gc.collect()

print('Memory usage reduction…')
%%time

bags = 3

models = []

for i in range(bags):

    print("Start training model {}".format(i))

    params['random_seed'] = i

    np.random.seed(42)

    random.seed(36)

    model = CatBoostRegressor(**params)

    model.fit(X_train, y_train, cat_features=categorical_indices, verbose=False)

    models.append(model)

    

# Sanity check (make sure scores on a small portion of the dataset are reasonable)

for i, model in enumerate(models):

    print("model {}: {}".format(i, abs(model.predict(X_val) - y_val).mean() * 100))
file_name = 'v23_EDA_catboost_ensemble_x3.csv.gz'

submission, pred_2016, pred_2017 = predict_and_export(models, prop2016, prop2017, file_name)
# %%time

# train_y = train['logerror'].values

# cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

# train_df = train.drop(['parcelid', 'logerror']+cat_cols, axis=1)

# feat_names = train_df.columns.values



# from sklearn import ensemble

# train_df_new=train_df.fillna(train_df.mean())

# model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)

# model.fit(train_df_new, train_y)



# ## plot the importance

# importances = model.feature_importances_

# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# indices = np.argsort(importances)[::-1][:20]



# plt.figure(figsize=(12,12))

# plt.title("Feature importances")

# plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

# plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

# plt.xlim([-1, len(indices)])

# plt.show()
# import xgboost as xgb

# xgb_params = {

#     'eta': 0.05,

#     'max_depth': 8,

#     'subsample': 0.7,

#     'colsample_bytree': 0.7,

#     'objective': 'reg:linear',

#     'silent': 1,

#     'seed' : 0

# }

# dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)

# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)



# # plot the important features #

# fig, ax = plt.subplots(figsize=(12,18))

# xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

# plt.show()