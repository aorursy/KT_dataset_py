import datetime

from datetime import datetime

from tqdm import tqdm

import pandas as pd

import numpy as np

import xgboost as xgb

import random

import zipfile

import time

import shutil

# Events

print('Read events...')

events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

appEvents = pd.read_csv("../input/app_events.csv")

appLabels = pd.read_csv("../input/app_labels.csv")

labelCategories = pd.read_csv("../input/label_categories.csv")



device = pd.DataFrame()

print(datetime.now())

#1. count of times used device

device['count'] = events.groupby(['device_id'])['event_id'].count()

print('Model features: events shape {}, appEvents shape {}, appLabels shape {}, labelCategories shape {}, device final shape {}'

.format(events.shape, appEvents.shape, appLabels.shape, labelCategories.shape, device.shape))



appLabels = pd.merge(appLabels, labelCategories, on='label_id')

appLabels = appLabels.drop('label_id',1) #there is no nulls here, this data is perfect

appEvents = appEvents[300000:600000] #forced to take a subset because too big...

appEvents = pd.merge(appEvents, appLabels, on='app_id')



#events = events[events.device_id == "-6401643145415154744"]

events = pd.merge(events, appEvents, how='left', on='event_id')

print(events.shape)

print(events.head(10))

print(events.isnull().sum()) #sometimes there is no array of apps with the event



#6. Number of active apps per device - How many apps do they use

eventsActiveApps = events[events.is_active == 1.0]

device['activeApps'] = eventsActiveApps.groupby(['device_id'])['is_active'].count()



#7. Number of installed apps per device - how many apps do they have

device['installedApps'] = events.groupby(['device_id'])['is_installed'].count()



#8. Top label across all apps - what types of apps do they get

print(events.groupby(['device_id'])['category'].value_counts())

device['topApp'] = events.groupby(['device_id'])['category'].nth(0)



print(device.head(50))



print(events.groupby(['device_id'])['category'].max())



#TODO: generate clusters of apps as well as top N apps/app clusters

# device['topLabel'] = events.groupby(['device_id'])['category'].top()

# device['topLabelCount'] = events.groupby(['device_id'])['category'].freq()



# eventsLabels = eventsLabels[eventsLabels.category != eventsLabels.top()]

# print(events.groupby(['device_id'])['category'].describe())



#9. Number of unique labels (how diverse are they)

device['uniqueLabelCount'] = events.groupby(['device_id'])['category'].unique()



print(events[events.device_id == "-1001384358977718793"]) #industry tag 36 112 count freq 22



print(device.head(50))



#10. top 5 apps used - what are they often using

#11. Labels amongst these top apps - what type do they often useprint(events.groupby(['device_id'])['category'].describe())



print(events.groupby(['device_id'])['category'].value_counts().index[0][1])

print(events.groupby(['device_id'])['category'].nth(0))



device['topApp'] = events.groupby(['device_id'])['category'].describe()[['top']]

print(device.head(50))

print(events.groupby(['device_id'])['category'].agg([np.sum, np.mean, np.std, len]))
print(events.groupby(['device_id'])['category'].describe())

print(events.groupby(['device_id'])['category'].describe().unstack())

print(events.groupby(['device_id','category']).size())

#print(events.groupby(['device_id'])['category'].value_counts().index[0][1])

print(events.groupby(['device_id'])['category'].nth(0))

#print(events.groupby(['device_id'])['category'].first())

print(events.groupby(['device_id','category'])['device_id'].agg({'Frequency':'count'}))



device['topApp'] = events.groupby(['device_id'])['category'].nth(0)

print(device.head(10))

#print(events.groupby(['device_id'])['category'].agg([np.sum, np.mean, np.std, len]))