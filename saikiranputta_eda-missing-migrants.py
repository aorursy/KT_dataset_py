import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")



%matplotlib inline
data = pd.read_csv("../input/MissingMigrantsProject.csv", encoding = "cp437")

data.head()
data.isnull().sum()
#Date features. 

data['date_day'] = pd.DatetimeIndex(data['date']).day

data['date_month'] = pd.DatetimeIndex(data['date']).month

data['date_year'] = pd.DatetimeIndex(data['date']).year



data['date_day'].value_counts().to_frame().plot(kind = "bar")
#That is 

# Monday - 0

# Tuesday - 1 etc. 

data['date_dayofweek'] = pd.DatetimeIndex(data['date']).dayofweek
data['date_dayofweek'].value_counts().plot(kind = "bar")
data['date_month'].value_counts().to_frame().plot(kind = "bar")
print(data['date_day'].isnull().sum())

print(data['date_day'].value_counts().head(5))





#subsituting with 20

data['date_day'] = data['date_day'].fillna(20.0)

print(data['date_day'].isnull().sum())
import math

data['week_number'] = [math.ceil(datum/7) for datum in data['date_day']]

data['week_number'].value_counts().plot(kind = 'bar')
#Lets do some data replacement. 

data['cause_of_death'] = data['cause_of_death'].fillna("Drowning")

data['cause_of_death'] = list(map(lambda string: string.lower(), data['cause_of_death']))

data['cause_of_death'].unique()
data['cause_of_death'].value_counts()
#Lets lessen the classes for ease. 

import re



def deathcause_replacement():

    global data

    data.loc[data['cause_of_death'].str.contains('sickness'), 'cause_of_death'] = 'sickness'

    data.loc[data['cause_of_death'].str.contains('harsh_weather'), 'cause_of_death'] = 'harsh_weather'

    data.loc[data['cause_of_death'].str.contains('unknown|unknow|north africa'), 'cause_of_death'] = 'unknown'

    data.loc[data['cause_of_death'].str.contains('starvation|dehydration'), 'cause_of_death'] = 'starvation'

    data.loc[data['cause_of_death'].str.contains('drowning|pulmonary|respiratory|lung|bronchial|pneumonia'), 'cause_of_death'] = 'drowning'

    data.loc[data['cause_of_death'].str.contains('hyperthermia'), 'cause_of_death'] = 'hypothermia'

    data.loc[data['cause_of_death'].str.contains('hypothermia'), 'cause_of_death'] = 'hypothermia'

    data.loc[data['cause_of_death'].str.contains('asphyxiation|suffocation'), 'cause_of_death'] = 'asphyxiation'

    data.loc[data['cause_of_death'].str.contains('train|bus|vehicle|truck|boat|car|road|van'), 'cause_of_death'] = 'vehicle accident'

    data.loc[data['cause_of_death'].str.contains('murder|stab|shot|violent|blunt force|violence|beat-up|fight|murdured|death'), 'cause_of_death'] = 'murder'

    data.loc[data['cause_of_death'].str.contains('crushed to death|crush'), 'cause_of_death'] = 'crushed'

    data.loc[data['cause_of_death'].str.contains('harsh conditions|harsh_weather'), 'cause_of_death'] = 'harsh conditions'

    data.loc[data['cause_of_death'].str.contains('diabetic|heart attack|sickness|meningitis|virus|cancer|bleeding|insuline|inhalation'), 'cause_of_death'] = 'health condition'

    data.loc[data['cause_of_death'].str.contains('electrocution'), 'cause_of_death'] = 'electrocution'
deathcause_replacement()



data['cause_of_death'].unique()
data['cause_of_death'].value_counts().plot(kind = "bar")
inspect = data['cause_of_death'].value_counts().to_frame().reset_index()

inspect.columns = ['cause_of_death', "death_count"]



name_list = inspect.loc[inspect['death_count'] >5 ]['cause_of_death'].tolist()
data = data.loc[data['cause_of_death'].isin(name_list)]
data['cause_of_death'].value_counts().plot(kind = "bar", title = "Reason for death")
[np.mean(data['lat']), np.mean(data['lon'])]
data['lon'] = data['lon'].fillna(np.mean(data['lon']))    #data['lon'][~np.isnan(data['lon'])].mean()

data['lat'] = data['lat'].fillna(np.mean(data['lat']))    #data['lon'][~np.isnan(data['lon'])].mean()
sns.factorplot(x = "lat", y = "lon", hue = "cause_of_death", kind = "swarm", data = data)
data['region_origin'].value_counts().plot(kind = "bar")
data['incident_region'].value_counts().plot(kind = "bar")
data.groupby('region_origin')['dead'].sum().to_frame().plot(kind = "bar")
#Lets change it a bit to make it more concrete!

data['region_origin'].unique()
data['region_origin'] = data['region_origin'].fillna('Africa')
data.loc[data['region_origin'].str.contains('Africa'), 'region_origin'] = 'Africa'

data.groupby('region_origin')['dead'].sum().to_frame().plot(kind = "bar")
data.groupby('region_origin')['dead'].sum().to_frame()

#Let's combine missing and dead. 

data['missing'].value_counts().head(10)
data['missing'] = data['missing'].fillna(1)

data['missing_and_dead'] = data['missing'] + data['dead']



data['missing_and_dead'].value_counts().head(10)
data.groupby('region_origin')['missing_and_dead'].sum().to_frame().plot(kind = "bar", stacked = True)
data['affected_nationality'].value_counts().head(15).plot(kind = "bar")