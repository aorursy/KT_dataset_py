# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing libraries



import os

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
#loading the dataset



data = pd.read_csv('../input/2008.csv')
data.head(5)
data.shape
#checking missing values



missing_data = data.isnull().sum(axis=0).reset_index()

missing_data.columns = ['variable', 'missing values']

missing_data['filling factor (%)']=(data.shape[0]-missing_data['missing values'])/data.shape[0]*100

missing_data.sort_values('filling factor (%)').reset_index(drop = True)
data['Date'] = pd.to_datetime(data.Year.map(str)+'-'+data.Month.map(str)+'-'+data.DayofMonth.map(str))
# Function that convert the 'HHMM' string to datetime.time

def format_heure(chaine):

    if pd.isnull(chaine):

        return np.nan

    else:

        if chaine == 2400: chaine = 0

        chaine = "{0:04d}".format(int(chaine))

        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))

        return heure



# Function that combines a date and time to produce a datetime.datetime

def combine_date_heure(x):

    if pd.isnull(x[0]) or pd.isnull(x[1]):

        return np.nan

    else:

        return datetime.datetime.combine(x[0],x[1])



# Function that combine two columns of the dataframe to create a datetime format

def create_flight_time(data, col):    

    liste = []

    for index, cols in data[['Date', col]].iterrows():    

        if pd.isnull(cols[1]):

            liste.append(np.nan)

        elif float(cols[1]) == 2400:

            cols[0] += datetime.timedelta(days=1)

            cols[1] = datetime.time(0,0)

            liste.append(combine_date_heure(cols))

        else:

            cols[1] = format_heure(cols[1])

            liste.append(combine_date_heure(cols))

    return pd.Series(liste)
data['CRSDepTime'] = create_flight_time(data, 'CRSDepTime')

data['DepTime'] = data['DepTime'].apply(format_heure)

data['CRSArrTime'] = data['CRSArrTime'].apply(format_heure)

data['ArrTime'] = data['ArrTime'].apply(format_heure)
data.loc[:5, ['CRSDepTime', 'CRSArrTime', 'DepTime',

             'ArrTime', 'DepDelay', 'ArrDelay']]
# Extracting statistical parameters from a groupby object:

def get_stats(group):

    return {'min': group.min(), 'max': group.max(),

            'count': group.count(), 'mean': group.mean()}



# Dataframe creation with statitical infos on each airline:

global_stats = data['DepDelay'].groupby(data['UniqueCarrier']).apply(get_stats).unstack()

global_stats = global_stats.sort_values('count')

global_stats
# Grouping the delays

delay_type = lambda x:((0,1)[x > 5],2)[x > 45]

data['DelayLvl'] = data['DepDelay'].apply(delay_type)



fig = plt.figure(1, figsize=(10,7))

ax = sns.countplot(y="UniqueCarrier", hue='DelayLvl', data=data)



# Setting Labels

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

ax.yaxis.label.set_visible(False)

plt.xlabel('Flight count', fontsize=16, weight = 'bold', labelpad=10)



# Setting Legends

L = plt.legend()

L.get_texts()[0].set_text('on time (t < 5 min)')

L.get_texts()[1].set_text('small delay (5 < t < 45 min)')

L.get_texts()[2].set_text('large delay (t > 45 min)')

plt.show()