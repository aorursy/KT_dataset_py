import os

import json

import pandas as pd

from pandas.io.json import json_normalize

import numpy as np

import matplotlib.pyplot as plt

import dateutil.parser



#from geopy.geocoders import Nominatim

#import geopandas as gp



%matplotlib inline
! head -40 ../input/drug-enforcement-0001-of-0001.json
! head -40 ../input/food-enforcement-0001-of-0001.json
! head -70 ../input/device-510k-0001-of-0001.json
food_file = open('../input/food-enforcement-0001-of-0001.json','r')

food_str = food_file.read()

food_json = json.loads(food_str)

food = json_normalize(food_json,'results')
drug_file = open('../input/drug-enforcement-0001-of-0001.json','r')

drug_str = drug_file.read()

drug_json = json.loads(drug_str)

drug = json_normalize(drug_json,'results')
device_file = open('../input/device-510k-0001-of-0001.json','r')

device_str = device_file.read()

device_json = json.loads(device_str)

device = json_normalize(device_json,'results')
print("device: "+str(device.shape))

print("food: "+str(food.shape))

print("drug: "+str(drug.shape))



c1 = pd.DataFrame({'device':device.columns.values}).sort_values(by='device')

c2 = pd.DataFrame({'food':food.columns.values}).sort_values(by='food')

c3 = pd.DataFrame({'drug':drug.columns.values}).sort_values(by='drug')



cols = pd.concat([c1,c2,c3], axis=1)

cols
enforce = drug.append(device, ignore_index=True)

enforce = enforce.append(food, ignore_index=True)

enforce.shape
enforce.product_type.value_counts(dropna=False).sort_index()
print("Drug recall counts: "+str(drug_json['meta']['results']['total']))

print("Food recall counts: "+str(food_json['meta']['results']['total']))

print("Device recall counts: "+str(device_json['meta']['results']['total']))
pd.options.display.max_columns = None

#drug.head(3)

print("Amount of recalls in dataset: "+str(enforce.shape[0]))
#drug.columns.values
enforce.isnull().sum()
(pd.DataFrame(drug).replace("",np.nan).isnull().sum())-drug.isnull().sum()
#Geographical Data

drug['geo_string']=drug['address_1']+", "+drug['city']+", "+drug['postal_code']+", "+drug['state']+", "+drug['country']



drug_geo = drug[['address_1', 'address_2','city','postal_code','state','country','geo_string']]

drug_geo.head()
#geolocator = Nominatim()



#for i in range(len(drug['geo_string'])):

#    location = geolocator.geocode(drug['geo_string'][i])

#    drug['geo_lat'][i] = location.latitude

#    drug['geo_lon'][i] = location.longitude
#Date & Time Info



#Parsing date info to datetime type

drug['recall_initiation_date'] = pd.to_datetime(drug['recall_initiation_date'])

drug['center_classification_date'] = pd.to_datetime(drug['center_classification_date'])

drug['report_date'] = pd.to_datetime(drug['report_date'])

drug['termination_date'] = pd.to_datetime(drug['termination_date'])



drug_date = drug[['recall_initiation_date','center_classification_date','report_date','termination_date']]

drug_date.head()
#pd.DataFrame(drug.groupby('country').count()['recall_number'])

drug.groupby('country').count()['recall_number']
drug[drug['country']=='']
drug[drug['recalling_firm']=='Pfizer Inc.'].groupby('address_1').count()['recall_number']
t_start = drug.recall_initiation_date.min()

t_end = drug.recall_initiation_date.max()



print(t_start)

print(t_end)
timeframe = t_end-t_start

print("Timeframe = "+str(round(timeframe.total_seconds()/60/60/24/365,2))+" years")
drug_date.head()
drug['total_duration_days']= (drug['termination_date']-drug['recall_initiation_date']).astype('timedelta64[D]')

drug_date = drug[['recall_initiation_date','center_classification_date','report_date','termination_date','total_duration_days']]

drug_date.head()
drug_date.isnull().sum()

#plt.hist(drug_date['total_duration_days'])
#Total duration statistics of recalls with a termination date

drug_date['total_duration_days'][drug_date['total_duration_days'].isnull() == False].describe()
#Percentage of recalls without termination date

int(round(drug_date['total_duration_days'].isnull().sum()/drug_date.shape[0]*10000))/100
bins = 30

plt.xlabel('Total recall duration (days)')

plt.ylabel('# Recalls (log-scale)')

plt.yscale('log')

plt.title('Distribution of the total recall duration')

plt.hist(drug_date['total_duration_days'][drug_date['total_duration_days'].isnull() == False], bins=bins)

plt.show()