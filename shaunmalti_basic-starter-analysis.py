import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
def explore(data):

    summaryDf = pd.DataFrame(data.dtypes, columns=['dtypes'])

    summaryDf = summaryDf.reset_index()

    summaryDf['Name'] = summaryDf['index']

    summaryDf['Missing'] = data.isnull().sum().values

    summaryDf['Total'] = data.count().values

    summaryDf['MissPerc'] = (summaryDf['Missing']/data.shape[0])*100

    summaryDf['NumUnique'] = data.nunique().values

    summaryDf['UniqueVals'] = [data[col].unique() for col in data.columns]

    print(summaryDf.head(30))
explore(data)
data.dtypes
sns.heatmap(data.corr())
sns.countplot(data.arrival_date_year)
data_july = data.loc[data.arrival_date_month=='July']

plt.figure(figsize=(15,6))

sns.countplot(data_july.arrival_date_day_of_month, hue=data_july.arrival_date_year)

plt.xticks(rotation=90)
pct_change = pd.DataFrame(data_july.groupby(['arrival_date_year'])['hotel'].count())

pct_change['pct_change'] = data_july.groupby(['arrival_date_year'])['hotel'].count().pct_change() * 100

pct_change
fig = plt.figure(figsize=(20,10))

plt.pie(data['country'].value_counts(), labels=data['country'].value_counts().index)

fig.set_facecolor('lightgrey')

plt.show()
df_bycountry = data_july.groupby(['country', 'arrival_date_year']).size().reset_index(name='counts')

plt.figure(figsize=(20,5))

sns.barplot(data=df_bycountry, x='country', y='counts', hue='arrival_date_year')

plt.ylabel('Count')

plt.xlabel('Country Code')

plt.xticks(rotation=90)
int_df = data_july.loc[data_july.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])]

df_by_country_hotel = int_df.groupby(['country', 'arrival_date_year', 'hotel']).size().reset_index(name='counts')
plt.figure(figsize=(20,10))

plt.subplot(2,1,1)

ax1 = sns.barplot(data=df_by_country_hotel.loc[df_by_country_hotel.hotel=='Resort Hotel'], x='country', y='counts', hue='arrival_date_year')

ax1.set_title('Resort Hotel')

plt.subplot(2,1,2)

ax2 = sns.barplot(data=df_by_country_hotel.loc[df_by_country_hotel.hotel=='City Hotel'], x='country', y='counts', hue='arrival_date_year')

ax2.set_title('City Hotel')
plt.figure(figsize=(20,6))

sns.countplot(data.arrival_date_week_number, hue=data.arrival_date_year)

plt.xticks(rotation=90)
data['lead_time'].describe()
sns.distplot(data['lead_time'])
data_time = data[['country', 'lead_time']]
data_nld = data_time.loc[data_time.country=='NLD']

data_usa = data_time.loc[data_time.country=='USA']

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

ax1=sns.distplot(data_nld['lead_time'])

ax1.set_title('lead time distribution - NLD')

plt.subplot(1,2,2)

ax2=sns.distplot(data_usa['lead_time'])

ax2.set_title('lead time distribution - USA')

plt.show()
data_cn = data_time.loc[data_time.country=='CN']

data_aut = data_time.loc[data_time.country=='AUT']

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

ax1=sns.distplot(data_cn['lead_time'])

ax1.set_title('lead time distribution - CN')

plt.subplot(1,2,2)

ax2=sns.distplot(data_usa['lead_time'])

ax2.set_title('lead time distribution - AUT')

plt.show()
labels = ['Resort Hotel Lead Time', 'City Hotel Lead Time']

plt.figure()

sns.kdeplot(data.loc[data.hotel=='Resort Hotel', 'lead_time'], shade=True)

sns.kdeplot(data.loc[data.hotel=='City Hotel', 'lead_time'], shade=True)

plt.legend(labels)
data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.groupby(['total_nights', 'hotel']).size().reset_index(name='counts')
data_tn = data.groupby(['total_nights', 'hotel']).size().reset_index(name='counts')

plt.figure(figsize=(15,5))

sns.barplot(x='total_nights', y='counts', hue='hotel', data=data_tn)

plt.xticks(rotation=90)

plt.show()
data_yr = data.loc[data.arrival_date_year==2016]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

data_yr_grp = data_yr.groupby(['arrival_date_month', 'hotel']).size().reset_index(name='counts')

plt.figure(figsize=(15,5))

sns.barplot(x='arrival_date_month', y='counts', hue='hotel', data=data_yr_grp, order=months)
data['kids'] = data.children + data.babies
data_kids = data.loc[data.kids>0]

ct1 = pd.crosstab(data_kids.kids, data_kids.hotel).apply(lambda x: x/x.sum(), axis=0)

data_babies = data.loc[data.babies>0]

ct2 = pd.crosstab(data_babies.kids, data_babies.hotel).apply(lambda x: x/x.sum(), axis=0)

ct1.plot.bar()
ct2.plot.bar()
data.loc[data.meal=='Undefined', 'meal'] = 'SC'
# taking top 10 seen countries

data_meal_tc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'meal']).size().reset_index(name='counts')

plt.figure(figsize=(20,5))

sns.barplot(data=data_meal_tc, x='country', y='counts', hue='meal')
perc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'meal']).size()

percbycountry = perc.groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).reset_index(name='percgp')
plt.figure(figsize=(20,5))

sns.barplot(data=percbycountry, x='country', y='percgp', hue='meal')
perc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'hotel']).size()

percbycountry = perc.groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).reset_index(name='percgp')
# looking at percentage from top 10 countries going to which hotel

plt.figure(figsize=(20,5))

sns.barplot(data=percbycountry, x='country', y='percgp', hue='hotel')
countries = ['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE']

x = 1

plt.figure(figsize=(20,10))

for country in countries:

    temp_df = data.loc[data.country==country].groupby(['hotel', 'meal']).size()

    perc = temp_df.groupby(level=0).apply(lambda x: x/float(x.sum()) * 100).reset_index(name='percgrp')

    plt.subplot(2, 5, x)

    ax = sns.barplot(data=perc, x='hotel', y='percgrp', hue='meal')

    ax.set_title(country)

    x+=1

plt.show()