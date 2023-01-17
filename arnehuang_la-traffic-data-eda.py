import numpy as np

import pandas as pd

import os

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('classic')



raw_data = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')

raw_data.info()
raw_data.head(5)
# get most and least common occuring items per field, up to 5

for column in raw_data.columns:

    uniques = raw_data[column].value_counts()

    print("{}\n{} distincts: {}".format(column, pd.concat([uniques.head(5), uniques.tail(5)]), len(uniques)))
# Create some features

time_labels = list(map(str, list(np.arange(0, 2400, 100))))

raw_data['Time Binned'] = pd.cut(raw_data['Time Occurred'], bins = 24, labels = time_labels)

raw_data['Reported Day'] = pd.to_datetime(raw_data['Date Reported']).dt.weekday

raw_data['Occurred Day'] = pd.to_datetime(raw_data['Date Occurred']).dt.weekday
# Take a look at timing of collisions, most occur after work with spike in mornings

plt.subplot(2, 1, 1)

sns.distplot(raw_data['Time Occurred'], bins=24)

v = raw_data['Time Binned'].value_counts(sort=False)

plt.subplot(2, 1, 2)

plt.bar(v.index, v.values)
# Area of collisions. Some areas 2x others.

raw_data.groupby('Area Name')['DR Number'].count().plot(kind='bar')
# heatmap of Area VS Time. Hollywood has a lot of accidents at night!

a_x_t = raw_data.pivot_table('DR Number', index='Time Binned', columns='Area Name', aggfunc='count')

sns.heatmap(a_x_t, cmap='Greys')
# Days of the week accidents occur on. Spike on Fridays.

raw_data.groupby('Occurred Day')['DR Number'].count().plot(kind='bar')
# There's a descrepancy between Occured day and Reported day, namely a lot of accidents happen on Friday and get reported later.

occurred_day = raw_data.groupby('Occurred Day')['DR Number'].count()

reported_day = raw_data.groupby('Reported Day')['DR Number'].count()

occurred_day.name = 'Occurred Day'

reported_day.name = 'Reported Day'

combined = pd.concat([occurred_day, reported_day], axis=1)

print(combined)

combined.plot()
# Males vs Female Accidents throughout the day. Note that there may just be more males on the roads then females. Or could it be that females are better drivers, except when going to work?

m_f = raw_data.pivot_table('DR Number', index='Time Binned', columns='Victim Sex', aggfunc='count')

m_f = m_f[['M', 'F']]

m_f.plot()
# Collisions by age. Note the spike at 99. Maybe the LAPD has some rule where undetermined ages get thrown into 99?

plt.hist(raw_data['Victim Age'], bins=100)

pass