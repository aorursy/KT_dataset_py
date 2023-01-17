import random

import datetime

import pandas as pd

import statistics

import numpy as np

import scipy

from scipy import stats



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.style as style

style.use('seaborn-pastel')

sns.set_style('whitegrid')

sns.set_context('paper')



%matplotlib inline
chicago = pd.read_csv('../input/Chicago-Divvy-2016.csv', 

                      parse_dates=['starttime', 'stoptime'],

                      dtype={'usertype': 'category',

                             'gender': 'category'})

chicago.head()
chicago.info()
data = chicago.sort_values(by='starttime')

data = data.reset_index()

print('Date rage of dataset: {} - {}'.format(data.loc[1, 'starttime'], data.loc[len(data)-1, 'stoptime']))
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)

groupby_user = data.groupby('usertype').size()

groupby_user.plot(kind='bar', title='Distribution of user types', rot=45);



plt.subplot(1,2,2)

groupby_gender = data.groupby('gender').size()

groupby_gender.plot(kind='bar', title='Distribution of genders', rot=45, color=['pink', 'lightblue']);
data = data.sort_values(by='birthyear')

groupby_birthyear = data.groupby('birthyear').size()

groupby_birthyear.plot(kind='bar', title='Distribution of birth years', figsize=(16,4));
data_mil = data.loc[(data['birthyear'] >= 1977) & (data['birthyear'] <= 1994)]

groupby_mil = data_mil.groupby('usertype').size()

groupby_mil.plot(kind='bar', title='Distribution of user types of millenials', 

                 rot=45);
groupby_birthyear_gender = data.groupby(['birthyear', 'gender'])['birthyear'].count().unstack('gender').fillna(0)

groupby_birthyear_gender.plot(kind='bar', title='Distribution of birth years by gender', 

                              stacked=True, figsize=(16,4), color=['pink', 'blue']);
groupby_birthyear_usertype = data.groupby(['birthyear',

                                           'usertype'])['birthyear'].count().unstack('usertype').fillna(0)

groupby_birthyear_usertype.plot(kind='bar', title='Distribution of birth years by user type',

                                stacked=True, figsize=(16,4));
data[data['usertype'] == 'Customer']['gender'].isnull().values.all()
data[data['usertype'] == 'Customer'].count()
data = data.set_index('starttime')
plt.figure(figsize=(16,12))



plt.subplot(2,2,1)

data.groupby(data.index.month)['tripduration'].count().plot.bar(

    title='Distribution of # trips by month', rot=0)

plt.xlabel('Month');

plt.ylabel('# trips');



plt.subplot(2,2,2)

data.groupby(data.index.day)['tripduration'].count().plot.bar(

    title='Distribution of # trips by day', rot=0)

plt.xlabel('Day');

plt.ylabel('# trips');



ax = plt.subplot(2,2,3)

data.groupby(data.index.weekday)['tripduration'].count().plot.bar(

    title='Distribution of # trips by day of the week', rot=0)

plt.xlabel('Day of the week');

plt.ylabel('# trips');

ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])





plt.subplot(2,2,4)

data.groupby(data.index.hour)['tripduration'].count().plot.bar(

    title='Distribution of # trips by hour', rot=0)

plt.xlabel('Hour');

plt.ylabel('# trips');