import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime 

import numpy as np
%%time

dtypes = {}

dtypes['MachineIdentifier'] = 'str'

dtypes['AvSigVersion'] = 'category'

dtypes['HasDetections'] = 'int8'



# LOAD TRAIN & TEST DATA

train = pd.read_csv('../input/microsoft-malware-prediction/train.csv', usecols=list(dtypes.keys()), dtype=dtypes)

test = pd.read_csv('../input/microsoft-malware-prediction/test.csv', usecols=list(dtypes.keys())[0:-1], dtype=dtypes)



# Load AvSigVersion Dates

dates1 = np.load('../input/avgsig/train_AvSigVersion.npy')[()]

dates2 = np.load('../input/avgsig/train_AvSigVersion2.npy')[()]

dates3 = np.load('../input/avgsig/test_AvSigVersion.npy')[()]
# process the dates, create a dictionary to store all dates

date = {}

for key, value in zip(dates1.keys(), dates1.values()):

    if key not in date.keys():

        date[key] = value

        

for key, value in zip(dates2.keys(), dates2.values()):

    if key not in date.keys():

        date[key] = value

        

for key, value in zip(dates3.keys(), dates3.values()):

    if key not in date.keys():

        date[key] = value
# function for stripping month, year, day, week data. try/except since there are missing dates

def strip_month(feature):

    try:

        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').month

    except: 

        return 0



def strip_year(feature):

    try:

        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').year

    except: 

        return 0



def strip_day(feature):

    try:

        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').day

    except: 

        return 0



def strip_week(feature):

    try:

        # be careful, there is a leap week. apparently there is a 53rd week!

        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').isocalendar()[1]

    except: 

        return 0



# binary featurization

def month11(feature):

    return 1 if feature == 11 else 0



def month10(feature):

    return 1 if feature == 10 else 0
%%time

# create a numerical feature that includes only months October/November

train['Month'] = train['AvSigVersion'].map(date).apply(strip_month)

test['Month'] = test['AvSigVersion'].map(date).apply(strip_month)

train['Year'] = train['AvSigVersion'].map(date).apply(strip_year)

test['Year'] = test['AvSigVersion'].map(date).apply(strip_year)

train['Day'] = train['AvSigVersion'].map(date).apply(strip_day)

test['Day'] = test['AvSigVersion'].map(date).apply(strip_day)

train['Week'] = train['AvSigVersion'].map(date).apply(strip_week)

test['Week'] = test['AvSigVersion'].map(date).apply(strip_week)



# binary features, specifically used to hack those months (for private LB)

train['AvSigMonth_10'] = train['Month'].apply(month10).astype('int8')

test['AvSigMonth_10'] = test['Month'].apply(month10).astype('int8')

train['AvSigMonth_11'] = train['Month'].apply(month11).astype('int8')

test['AvSigMonth_11'] = test['Month'].apply(month11).astype('int8')
# Explore Detection Counts by Month

plt.figure(figsize=(16,8))

sns.countplot(train['Month'], hue=train['HasDetections'])
plt.figure(figsize=(16,8))

sns.countplot(test['Month'])
# Zoom into detection counts for Nov - Dec

subset_train = train[train['Month'] >= 10]

sns.countplot(subset_train['Month'], hue=subset_train['HasDetections'])