import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

from collections import Counter

import numpy as np
data_dir = '../input/'



def load_csv(f_name, dtype=None, parse_dates=[]):

    return pd.read_csv(data_dir + f_name + '.csv', dtype=dtype, parse_dates=parse_dates)



air_reserve = load_csv('air_reserve', {

                                        'air_store_id': 'category',

                                        'visit_datetime': str,

                                        'reserve_datetime': str,

                                        'reserve_visitors': int},

                       ['visit_datetime', 'reserve_datetime']

                      )

hpg_reserve = load_csv('hpg_reserve', {

                                        'hpg_store_id': 'category',

                                        'visit_datetime': str,

                                        'reserve_datetime': str,

                                        'reserve_visitors': int},

                       ['visit_datetime', 'reserve_datetime']

                      )

air_store_info = load_csv('air_store_info', {

    'air_store_id': 'category',

    'air_genre_name': 'category',

    'air_area_name': 'category',

    'latitude': float,

    'longitude': float                                        

})

hpg_store_info = load_csv('hpg_store_info', {

    'hpg_store_id': 'category',

    'hpg_genre_name': 'category',

    'hpg_area_name': 'category',

    'latitude': float,

    'longitude': float                                        

})

store_id_relation = load_csv('store_id_relation', {

    'hpg_store_id': 'category',

    'air_store_id': 'category'

})

air_visit_data = load_csv('air_visit_data', {

    'air_store_id': 'category',

    'visit_date': str,

    'visitors': int

}, ['visit_date'])

date_info = load_csv('date_info', {

    'calendar_date': str,

    'day_of_week': 'category',

    'holiday_flg': bool

}, ['calendar_date'])
air_reserve.head()
print('Number of entries:', len(air_reserve))

for col_name in air_reserve:

    print('Number of null entries in {}: {}'.format(col_name, air_reserve[col_name].isnull().sum()))

print('Number of unique store ids:', len(air_reserve['air_store_id'].unique()))
plt.hist(air_reserve.reserve_visitors)

plt.title('Number of visitors in reservation')

plt.show()



visit_hours = [visit_time.to_pydatetime().hour for visit_time in air_reserve.visit_datetime]

plt.hist(visit_hours)

plt.title('Most popular visiting hours')

plt.show()



reserve_hours = [reserve_time.to_pydatetime().hour for reserve_time in air_reserve.reserve_datetime]

plt.hist(reserve_hours)

plt.title('Most popular reservation hours')

plt.show()
years = matplotlib.dates.YearLocator()

months = matplotlib.dates.MonthLocator()  

yearsFmt = matplotlib.dates.DateFormatter('%Y')



plt.plot_date(air_reserve.reserve_datetime, air_reserve.reserve_visitors)

plt.xlabel('Reserve time')

plt.ylabel('Number of reservation visitors')

ax = plt.gca()

ax.xaxis.set_major_locator(years)

ax.xaxis.set_major_formatter(yearsFmt)

ax.xaxis.set_minor_locator(months)

plt.show()
time_diffs = []



for (reserve, visit) in zip(air_reserve.reserve_datetime, air_reserve.visit_datetime):

    diff = visit.to_pydatetime() - reserve.to_pydatetime()

    time_diffs.append(diff.seconds / 3600) # hours

    

plt.hist(time_diffs)

plt.title('Hours between reservation and visit')

plt.show()
hpg_reserve.head()
print('Number of entries:', len(hpg_reserve))

for col_name in hpg_reserve:

    print('Number of null entries in {}: {}'.format(col_name, hpg_reserve[col_name].isnull().sum()))

print('Number of unique store ids:', len(hpg_reserve['hpg_store_id'].unique()))
plt.hist(hpg_reserve.reserve_visitors)

plt.title('Number of visitors in reservation')

plt.show()



visit_hours = [visit_time.to_pydatetime().hour for visit_time in hpg_reserve.visit_datetime]

plt.hist(visit_hours)

plt.title('Most popular visiting hours')

plt.show()



reserve_hours = [reserve_time.to_pydatetime().hour for reserve_time in hpg_reserve.reserve_datetime]

plt.hist(reserve_hours)

plt.title('Most popular reservation hours')

plt.show()
years = matplotlib.dates.YearLocator()

months = matplotlib.dates.MonthLocator()  

yearsFmt = matplotlib.dates.DateFormatter('%Y')



plt.plot_date(air_reserve.reserve_datetime, air_reserve.reserve_visitors)

plt.xlabel('Reserve time')

plt.ylabel('Number of reservation visitors')

ax = plt.gca()

ax.xaxis.set_major_locator(years)

ax.xaxis.set_major_formatter(yearsFmt)

ax.xaxis.set_minor_locator(months)

plt.show()
time_diffs = []



for (reserve, visit) in zip(hpg_reserve.reserve_datetime, hpg_reserve.visit_datetime):

    diff = visit.to_pydatetime() - reserve.to_pydatetime()

    time_diffs.append(diff.seconds / 3600) # hours

    

plt.hist(time_diffs)

plt.title('Hours between reservation and visit')

plt.show()
air_store_info.head()
print('Number of entries:', len(air_store_info))

for col_name in air_store_info:

    print('Number of null entries in {}: {}'.format(col_name, air_store_info[col_name].isnull().sum()))

print('Number of unqiue genre names:', len(air_store_info.air_genre_name.unique()))

print('Number of unique area names:', len(air_store_info.air_area_name.unique()))
counts = Counter(air_store_info.air_genre_name)

common = counts.most_common()

labels = [item[0] for item in common]

number = [item[1] for item in common]



plt.bar(range(len(common)), number, tick_label=labels)

plt.title('Genre distribution')

plt.xlabel('Genre name')

plt.ylabel('Number of restaurants with genre')

plt.xticks(rotation=90)

plt.show()
counts = Counter(air_store_info.air_area_name)

common = counts.most_common(15)

labels = [item[0] for item in common]

number = [item[1] for item in common]



plt.bar(range(len(common)), number, tick_label=labels)

plt.title('Top 15 area names')

plt.xlabel('Area name')

plt.ylabel('Number of restaurants in area')

plt.xticks(rotation=90)

plt.show()
hpg_store_info.head()
print('Number of entries:', len(hpg_store_info))

for col_name in hpg_store_info:

    print('Number of null entries in {}: {}'.format(col_name, hpg_store_info[col_name].isnull().sum()))

print('Number of unqiue genre names:', len(hpg_store_info.hpg_genre_name.unique()))

print('Number of unique area names:', len(hpg_store_info.hpg_area_name.unique()))
counts = Counter(hpg_store_info.hpg_genre_name)

common = counts.most_common()

labels = [item[0] for item in common]

number = [item[1] for item in common]



plt.bar(range(len(common)), number, tick_label=labels)

plt.title('Genre distribution')

plt.xlabel('Genre name')

plt.ylabel('Number of restaurants with genre')

plt.xticks(rotation=90)

plt.show()
counts = Counter(hpg_store_info.hpg_area_name)

common = counts.most_common(15)

labels = [item[0] for item in common]

number = [item[1] for item in common]



plt.bar(range(len(common)), number, tick_label=labels)

plt.title('Top 15 area names')

plt.xlabel('Area name')

plt.ylabel('Number of restaurants in area')

plt.xticks(rotation=90)

plt.show()
store_id_relation.head()
print('Number of entries:', len(store_id_relation))

print('Percentage of AIR store ids with relation: %.2lf%%' % ((len(store_id_relation) / len(air_store_info) * 100)))

print('Percentage of HPG store ids with relation: %.2lf%%' % ((len(store_id_relation) / len(hpg_store_info) * 100)))
air_visit_data.head()
date_info.head()
print('Total number of holidays', sum(date_info.holiday_flg))
print('All Holidays:')

list(date_info[date_info.holiday_flg == 1].calendar_date.dt.date)
train_merged = pd.merge(air_visit_data, store_id_relation, on='air_store_id', how='left') # Will result in a lot of NaN values

train_merged = pd.merge(train_merged, date_info, left_on='visit_date', right_on='calendar_date')

train_merged = pd.merge(train_merged, air_store_info, on='air_store_id')
train_merged.head()
holidays = train_merged[train_merged.holiday_flg == True]

non_holidays = train_merged[train_merged.holiday_flg == False]



print('Average number of visitors on holidays:', np.mean(holidays.visitors))

print('Average number of visitors on normal days:', np.mean(non_holidays.visitors))