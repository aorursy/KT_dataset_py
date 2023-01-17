# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import time

import random

from math import *

import operator

import pandas as pd

import numpy as np



# import plotting libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

#from pandas.plotting import scatter_matrix

%matplotlib inline

#plt.style.use(['fivethirtyeight'])

#mpl.rcParams['lines.linewidth'] = 2



import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

sns.set(rc={'figure.figsize':(12,7)})

sns.set(style='white')

patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

route = pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')

time_series = pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')
patient.head()
patient.info()
column = ['sex', 'country', 'region', 'group',

       'infection_reason', 'infection_order', 

       'state']

for col in column:

    print(patient[col].value_counts())

    print('\n')
plt.figure(figsize=(12,7))

area = list(patient['region'].value_counts().sort_values(ascending=False).index)

region_data = patient.region.value_counts().rename_axis('region').reset_index(name='count')

sns.barplot(x='count', y='region', order=area, data=region_data, palette='gist_gray')

plt.title('Regionwise confirmed cases', fontsize=15)

plt.show()
plt.figure(figsize=(12,7))

sns.countplot(patient.state, palette='gist_gray')

plt.yscale('log')

plt.title('condition of reported cases', fontsize=15)

plt.show()
plt.figure(figsize=(12,7))

sns.countplot(patient.sex, palette='gist_gray',hue=patient.state)

#plt.yscale('log')

plt.title('Regionwise confirmed cases by gender', fontsize=15)

plt.show()
plt.figure(figsize=(12,7))

reason_list = list(patient.infection_reason.value_counts().sort_values(ascending=False).index)

reason_data = patient.infection_reason.value_counts().rename_axis('reason').reset_index(name='count')

sns.barplot(x='count', y='reason', order=reason_list, data=reason_data, palette='gist_gray')

plt.title('Reason of Infection', fontsize=15)

plt.show()
patient['age'] = 2020 - patient.birth_year
def age_grp(age):

    if age > 0:

        if age%10 != 0:

            lower = int(floor(age/10)*10)

            upper = int(ceil(age/10)*10)-1

            return '{}-{}'.format(lower, upper)

        else:

            lower = int(age)

            upper = int(age)+9

            return '{}-{}'.format(lower,upper)

    else:

        return np.nan
patient['age_group'] = patient.age.apply(age_grp)
# Plotting age of affected peaple

sns.set(rc={'figure.figsize':(20,7)})

sns.set(style='white')

sns.countplot(patient.age.dropna().astype('int64'), orient='h', palette='viridis')

plt.title('Confirmed cases by age', fontsize=15)

plt.show()
# Plotting age of affected peaple

sns.set(rc={'figure.figsize':(20,7)})

sns.set(style='white')

sns.countplot(patient.age_group.dropna().sort_values(ascending=True), orient='h', palette='viridis')

plt.title('Confirmed cases by age group', fontsize=15)

plt.show()
# Parsing dates

import datetime as dt

patient.confirmed_date = pd.to_datetime(patient.confirmed_date)

patient.released_date = pd.to_datetime(patient.released_date)

patient.deceased_date = pd.to_datetime(patient.deceased_date)
# Extracting days to recovery and days to death

patient['days_to_recovery'] =  abs((patient.released_date - patient.confirmed_date).dt.days)

patient['days_to_death'] =  abs((patient.deceased_date - patient.confirmed_date).dt.days)
# Seperate dataframes for recovery and death data

data_recovery = patient.days_to_recovery.value_counts().sort_index().rename_axis('days_to_recovery').reset_index(name='Count')

data_death = patient.days_to_death.value_counts().sort_index().rename_axis('days_to_death').reset_index(name='Count')
# Changing the datatype of number of days to recovery and death

data_recovery.days_to_recovery = data_recovery.days_to_recovery.astype('int64').astype('object')

data_death.days_to_death = data_death.days_to_death.astype('int64').astype('object')
# Plotting the frequency of days to death

sns.barplot(x='days_to_recovery', y='Count', data = data_recovery, palette='viridis', orient='v')

plt.title('frequency of days taken for recovery', fontsize=15)

plt.show()
# Plotting the frequency of days to death

sns.barplot(x='days_to_death', y='Count', data = data_death, palette='viridis', orient='v')

plt.title('frequency of days until death', fontsize=15)

plt.show()
recovery_data = patient.groupby(['age_group', 'sex'])['days_to_recovery'].mean().dropna().reset_index().rename_axis()

ax = sns.lineplot(x='age_group', y='days_to_recovery',hue='sex',style='sex', data=recovery_data, palette='gist_gray_r')

plt.title('Days taken for recovery by age group', fontsize=15)

plt.show()
death_data = patient.groupby(['age_group', 'sex'])['days_to_death'].mean().dropna().reset_index().rename_axis()

ax = sns.lineplot(x='age_group', y='days_to_death',hue='sex',style='sex', data=death_data, palette='gist_gray_r')

plt.title('Days until death by age group', fontsize=15)

plt.show()
time_series = time_series.set_index('date')

time_series.head()
#mpl.style.use('fivethirtyeight')

sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(16,7))

ax.plot(time_series['acc_test'], label='test', linestyle='dashed', color='#FFC300', linewidth=3, markersize=6)

ax.plot(time_series['acc_negative'], label='negative', linestyle='dotted', color='#B6FF33', linewidth=3, markersize=6)

ax.plot(time_series['acc_confirmed'], label='confirmed', color='#FF5733', linewidth=4, markersize=6)

plt.title("Cases over time", fontsize=16)

#ax.set_xticks()

#plt.yscale('log')

ax.set_xticklabels(time_series.index, rotation=45)

plt.xlabel('Date', fontsize=16)

plt.ylabel('Number of Cases', fontsize=16)

plt.legend(loc='upper left', fontsize=15, fancybox=True, ncol=3, shadow=True)

plt.show()
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='cartodbpositron')



for lat, lon,city in zip(route['latitude'], route['longitude'], route['city']):

    folium.CircleMarker([lat, lon],

                        radius=4,

                        color='black',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='black',

                        fill_opacity= 0.2).add_to(southkorea_map)

southkorea_map