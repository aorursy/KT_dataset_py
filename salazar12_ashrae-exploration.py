# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder 

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler,LabelEncoder



import gc

import psutil



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")



sns.set()

msno.matrix(building_metadata)

plt.show()

building_metadata.head()
print("lenght: ", len(building_metadata))

#for numeric features

building_metadata[['square_feet','floor_count']].describe()
#categorical features

print(building_metadata['primary_use'].value_counts())

print(building_metadata['site_id'].value_counts())



#dates

building_metadata.year_built.dropna().sort_values()
print(len(train))
msno.matrix(train)

plt.show()

train.head()
#numeric varialbes

train[['meter_reading']].describe()

#categorial

print(train['meter'].value_counts())

#dates

train.timestamp.drop_duplicates().sort_values()
print('length: ', len(weather_train))
msno.matrix(weather_train)

plt.show()

weather_train.head()
#numeric varialbes

weather_train[['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

                    'sea_level_pressure','wind_direction','wind_speed']].describe()
#dates

weather_train.timestamp.sort_values().drop_duplicates()
pd_data = train.merge(building_metadata, how='left',on='building_id')

pd_data = pd_data.merge(weather_train, how = 'left', on = ['site_id','timestamp'])
print(len(train))

print(len(building_metadata))

print(len(weather_train))

print(len(pd_data))



pd_data.isna().sum()
num_features = ['floor_count','year_built','square_feet','air_temperature','cloud_coverage',

               'dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction'

               ,'wind_speed']

cat_features = ['primary_use','site_id','meter']

target = ['meter_reading']
print(pd_data[num_features].dtypes)

for col in pd_data.columns:

    if(pd_data[col].dtype == np.float64):

        pd_data[col] = pd_data[col].astype(np.float32)

    if(pd_data[col].dtype==np.int64):

        pd_data[col] = pd_data[col].astype(np.int32)

        

print("available RAM:", psutil.virtual_memory())



gc.collect()



print("available RAM:", psutil.virtual_memory())
print(pd_data[num_features].dtypes)

print(pd_data['meter_reading'].describe())

plt.figure(figsize=(15,7))

#with respect to meter

plt.subplot(1,2,1)

sns.boxplot(y = pd_data['meter_reading'])

plt.xlabel('meter_reading')



plt.subplot(1,2,2)

sns.boxplot(y = pd_data['meter_reading'],showfliers=False)

plt.xlabel('meter_reading')

# site population

dd = building_metadata.site_id.value_counts().reset_index().rename(columns={'index':'site','site_id':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'site',y='count',data = dd)

plt.title('population for each site')
# meter population

dd = pd_data.meter.value_counts().reset_index().rename(columns={'index':'meter_type','meter':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'meter_type',y='count',data = dd)

plt.title('population for each type of meter')
#primary use

dd = building_metadata.primary_use.value_counts().reset_index().rename(columns={'index':'primary_use','primary_use':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'primary_use',y='count',data = dd)

plt.xticks(rotation=90);

plt.title('population for use')
print('year built mode: ', building_metadata['year_built'].dropna().mode())

# year built

plt.figure(figsize=(11,7))

sns.distplot(building_metadata['year_built'].dropna(),kde=True)

plt.ylabel('occurences')

plt.xlabel('year built')

plt.title('distribution of year built')

plt.show()
#we can look at how the meter reading changes during the year

pd_stamp = pd_data.copy()

pd_stamp['timestamp'] = pd.to_datetime(pd_stamp.timestamp)

pd_stamp['timestamp'] = pd_stamp.timestamp.dt.date

pd_meter = pd_stamp.fillna(0).groupby(['site_id','timestamp'])['meter_reading'].mean().reset_index()

#pd_weather.isna().sum()

pd_meter = pd_meter.set_index('timestamp')
#meter reading

plt.figure(figsize=(12,35))

i=1

for s in pd_meter.site_id.unique():

    plt.subplot(16,1,i)

    plt.plot(pd_meter[pd_meter['site_id']==s]['meter_reading'],alpha=0.5, color='navy',label=str(s));

    i+=1

    plt.legend() 

plt.tight_layout()
pd_meter = pd_stamp.fillna(0).groupby(['site_id','meter','timestamp'])['meter_reading'].mean().reset_index()

pd_weather.isna().sum()

pd_meter = pd_meter.set_index('timestamp')

#meter reading

plt.figure(figsize=(12,35))

i=1

for s in pd_meter.site_id.unique():

    plt.subplot(16,1,i)

    for j in range(0,len(pd_meter.meter.unique())):

        me = pd_meter.meter.unique()[j] 

        plt.plot(pd_meter[(pd_meter['site_id']==s) & (pd_meter['meter']==me)]['meter_reading'],alpha=0.5,label='site_'+str(s)+'_meter_'+str(me));

    i+=1

    plt.legend() 

plt.tight_layout()
print(pd_data['meter_reading'].describe())



plt.figure(figsize=(11,7))

sns.distplot(pd_data['meter_reading'],kde=False)

plt.ylabel('occurences')

plt.xlabel('meter reading')

plt.title('all meter readings')

plt.show()





plt.figure(figsize=(11,7))

sns.distplot(pd_data[pd_data['meter_reading']<=(10**3)]['meter_reading'],kde=False)

plt.ylabel('occurences')

plt.xlabel('meter reading')

plt.title('meters below Q3')

plt.show()



plt.figure(figsize=(11,7))

sns.distplot(pd_data[(pd_data['meter_reading']>(10**3)) & (pd_data['meter_reading']<=(10**5))]['meter_reading'],kde=False)

plt.ylabel('occurences')

plt.xlabel('meter reading')

plt.title('Above 10^3 and below in 10^5')

plt.show()



plt.figure(figsize=(11,7))

sns.distplot(pd_data[pd_data['meter_reading'] >(10**5)]['meter_reading'],kde=False)

plt.ylabel('occurences')

plt.xlabel('meter reading')

plt.title('Above 10^5')

plt.show()
pd_outliers = pd_data[pd_data['meter_reading']>(10**4)]

pd_outlier_buildings = pd_outliers.drop_duplicates(subset='building_id',keep='first')

print("total number of buildings: ",len(pd_data.building_id.unique()) )

print("number of buildings with more than 10^4 in meter reading: ",len(pd_outlier_buildings) )

len(pd_data.building_id.unique())


msno.matrix(pd_outliers)

plt.show()
# outlier population by site

dd = pd_outlier_buildings.site_id.value_counts().reset_index().rename(columns={'index':'site','site_id':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'site',y='count',data = dd)

plt.title('outleir population for each site')
# meter population

dd = pd_outliers.meter.value_counts().reset_index().rename(columns={'index':'meter_type','meter':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'meter_type',y='count',data = dd)

plt.title('outlier population for each type of meter')
b_plot = pd_data[pd_data.site_id.isin([9,13])]['meter'].value_counts().reset_index().rename(columns={'index':'meter_type',

                                                                                           'meter':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'meter_type',y='count',data = b_plot)

plt.title('meter types in sites 9 and 13')
pd_stamp = pd_data.copy()

pd_stamp['timestamp'] = pd.to_datetime(pd_stamp.timestamp)

pd_stamp['timestamp'] = pd_stamp.timestamp.dt.date

pd_weather = pd_stamp.fillna(-1).groupby(['site_id','timestamp']).agg({'air_temperature':['mean', 'max','min'], 

                            'cloud_coverage':['mean', 'max','min'],'dew_temperature':['mean', 'max','min'],

                            'precip_depth_1_hr':['mean', 'max','min'],'sea_level_pressure':['mean', 'max','min'],

                            'wind_direction':['mean', 'max','min']}).reset_index()

#pd_weather.isna().sum()

pd_weather = pd_weather.set_index('timestamp')
pd_stamp = pd_data.copy()

pd_stamp['timestamp'] = pd.to_datetime(pd_stamp.timestamp)

pd_stamp['timestamp'] = pd_stamp.timestamp.dt.date
#for air temperature

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['mean'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['mean'],alpha=0.2, color='navy');



plt.title('mean air temperature, outlier sites in red')

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['max'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['max'],alpha=0.2, color='navy');



plt.title('max air temperature, outlier sites in red')

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['min'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['air_temperature']['min'],alpha=0.2, color='navy');



plt.title('min air temperature, outlier sites in red')

#for cloud cover

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['mean'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['mean'],alpha=0.2, color='navy');



plt.title('mean cloud coverage, outlier sites in red')
plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['max'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['max'],alpha=0.2, color='navy');



plt.title('max cloud coverage, outlier sites in red')
plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['min'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['cloud_coverage']['min'],alpha=0.2, color='navy');



plt.title('min coverage, outlier sites in red')
#fordew

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['mean'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['mean'],alpha=0.2, color='navy');



plt.title('mean dew_temperature, outlier sites in red')


#for dew

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['max'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['max'],alpha=0.2, color='navy');



plt.title('max dew_temperature, outlier sites in red')
plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['min'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['dew_temperature']['min'],alpha=0.2, color='navy');



plt.title('min dew_temperature, outlier sites in red')
#precipitation depth

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['mean'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['mean'],alpha=0.2, color='navy');



plt.title('mean precip_depth_1_hr, outlier sites in red')
#precipitation depth

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['min'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['min'],alpha=0.2, color='navy');



plt.title('min precip_depth_1_hr, outlier sites in red')
#precipitation depth

plt.figure(figsize=(15,7))

for s in pd_weather.site_id.unique():

    if(s==13 or s==9):

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['max'],color='red');

    else:

        plt.plot(pd_weather[pd_weather.site_id==s]['precip_depth_1_hr']['max'],alpha=0.2, color='navy');



plt.title('max precip_depth_1_hr, outlier sites in red')
#primary use

dd = pd_outlier_buildings.primary_use.value_counts().reset_index().rename(columns={'index':'primary_use','primary_use':'count'})

plt.figure(figsize=(10,6))

sns.barplot(x = 'primary_use',y='count',data = dd)

plt.xticks(rotation=90);

plt.title('population for use')
print('year built mode: ', pd_outlier_buildings['year_built'].dropna().mode())

print('buildings with the data available ', len(pd_outlier_buildings['year_built'].dropna().unique()))



# year built

plt.figure(figsize=(11,7))

sns.distplot(pd_outlier_buildings['year_built'].dropna(),kde=False)

plt.ylabel('occurences')

plt.xlabel('year built')

plt.title('distribution of year built')

plt.show()
#correlation matrix of numerical features

corrmat = pd_data[target+num_features].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# correlation of categorical variables to meter

plt.figure(figsize=(15,8))

ax = sns.boxplot(x='primary_use', y='meter_reading', data=pd_data);

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.figure(figsize=(15,8))

ax = sns.boxplot(x='primary_use', y='meter_reading', data=pd_data,showfliers = False);

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
pd_data_in_built = pd_data.copy()

#pd_data_in_built['year_built'] = pd.to_datetime(pd_data_in_built['year_built']).dt.year

pd_data_in_built = pd_data_in_built.sort_values(by='year_built')



pd_data_in_built.dropna(inplace=True)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year_built', y="meter_reading", data=pd_data_in_built)

plt.xticks(rotation=90);
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year_built', y="meter_reading", data=pd_data_in_built,showfliers=False)

plt.xticks(rotation=90);
weather_cols = ['meter_reading','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

               'sea_level_pressure','wind_direction','wind_speed']



pd_stamp= pd_data.copy()

pd_stamp['timestamp'] = pd.to_datetime(pd_stamp.timestamp)

pd_stamp['timestamp'] = pd_stamp.timestamp.dt.date

pd_w= pd_stamp.fillna(0).groupby(['building_id','timestamp'])[weather_cols].mean().reset_index()

plt.figure(figsize=(30,30))

g = sns.pairplot(pd_w[weather_cols])

plt.show() 
print("available RAM:", psutil.virtual_memory())



gc.collect()



print("available RAM:", psutil.virtual_memory())
pd_building = pd_data.groupby('building_id')['meter_reading'].mean().reset_index()

m_cols = ['building_id','year_built','floor_count','square_feet','meter','primary_use','site_id']

pd_building = pd_building.merge(pd_data[m_cols], on='building_id',how='left').sort_values(by='year_built')

pd_building.drop_duplicates(inplace=True)
#year built to meter reading

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year_built', y="meter_reading", data=pd_building,showfliers=True)

plt.xticks(rotation=90);

plt.show()
#year built to size

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year_built', y="square_feet", data=pd_building,showfliers=True)

plt.xticks(rotation=90);

plt.show()



#year built to size

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year_built', y="square_feet", data=pd_building,showfliers=False)

plt.xticks(rotation=90);

plt.show()

#year built to size

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.scatterplot(x='year_built', y="square_feet", data=pd_building,hue='meter')

plt.xticks(rotation=90);

plt.show()

#year built to site

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.violinplot(x='site_id', y="year_built", data=pd_building)

plt.xticks(rotation=90);

plt.show()



#year built to site and meter

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.violinplot(x='site_id', y="year_built", data=pd_building, hue='meter')

plt.xticks(rotation=90);

plt.show()


print("available RAM:", psutil.virtual_memory())



gc.collect()



print("available RAM:", psutil.virtual_memory())