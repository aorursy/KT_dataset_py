# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# to waork with date time data

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

bikesharing_data = pd.read_csv('../input/bike-share-daily-data/bike_sharing_daily.csv')

bikesharing_data.head(10)
# convert to a suitable date time format

bikesharing_data['dteday'] = pd.DatetimeIndex(bikesharing_data['dteday'])
# plot the data

plt.figure(figsize=(20,8))

plt.plot(bikesharing_data['dteday'],bikesharing_data['registered']
         ,color='b',label='registered')

plt.plot(bikesharing_data['dteday'],bikesharing_data['casual']
         ,color='r',label='casual')


plt.legend(loc='upper left')

plt.title('Bike Sharing Users')
plt.xlabel('Date')
plt.ylabel('counts of Bike Rentals')

plt.show()


year_df = bikesharing_data.groupby('yr', as_index=False).mean()

year_df[['yr','cnt']]
plt.figure(figsize=(12,8))

colors = ['b','m']

plt.bar(year_df['yr'],year_df['cnt'], width= 0.2, color = colors)


plt.xticks([0,1],['2011',2012])

plt.title('Bike Sharing Daily');
plt.xlabel('year')
plt.ylabel('mean count')

plt.show()

days = bikesharing_data.groupby('workingday', as_index=False).mean()

days['cnt']
plt.figure(figsize=(12,8))

colors = ['red','limegreen']

plt.bar(days['workingday'],days['cnt'], width= 0.2, color = colors)


plt.xticks([0,1],['Holiday','Working day'])

plt.title('Bike Sharing Daily');
plt.xlabel('Days')
plt.ylabel('Average Count of Rental Bikes')

plt.show()
year_data = bikesharing_data.loc[bikesharing_data['yr']==1]

year_data.head()

# lets see if there monthly variations across 2012 year

month_df = year_data[['mnth','cnt']].groupby('mnth',as_index=False).mean()

month_df
# lets show this monthly data as bars

# replace month numbers by month names

month_df['mnth'].replace([1,2,3,4,5,6,7,8,9,10,11,12],['jan','Feb','Mar'
                                                   ,'Apr','May','Jun',
                                                   'Jul','Aug','Sep',
                                                   'Oct', 'Nov','Dec'],inplace=True)
month_df
plt.figure(figsize=(12,8))

colors = ['b','g','c','r','m','y','k','c']

plt.bar(month_df['mnth'],month_df['cnt'], color = colors)


plt.title('Bike Sharing Daily');
plt.xlabel('Months')
plt.ylabel('Average Count of Bikes Rental')

plt.show()
#----------------------- Box Plots Violion Plots and Pie Charts
plt.figure(figsize=(12,8))

colors = ['b','g','c','r','m','y','k','c']

plt.boxplot(bikesharing_data['cnt'])


plt.xticks([1],['Rental Bikes'])

plt.title('Bike Sharing Daily');
plt.xlabel('Bike Sharing Daily')
plt.ylabel('Total Counts of Rental Bikes')

plt.show()
selected_bike_data = bikesharing_data[['casual','registered']]

selected_bike_data.head()
columns = selected_bike_data.columns

columns
bike_data_array = selected_bike_data.values

bike_data_array
plt.figure(figsize=(12,8))

bp = plt.boxplot(bike_data_array,
                 patch_artist=True,
                 labels=columns)

for i in range(len(bp['boxes'])):
    bp['boxes'][i].set(facecolor=colors[i])

plt.title('Bike Sharing Users');
plt.xlabel('Users')
plt.ylabel('Counts of Bikes Rental')

plt.show()
colors = ['g','m']

plt.figure(figsize=(12,8))


vp = plt.violinplot(bike_data_array)

plt.xticks([1,2],columns)

plt.title('Bike Sharing Users');
plt.xlabel('Users')
plt.ylabel('Counts of Bikes Rental')

plt.show()
season_data = bikesharing_data[['season','cnt']]

season_data.sample(5)


grouped_data = season_data.groupby('season',as_index=False).sum()

grouped_data.head()

grouped_data['season'].replace([1,2,3,4],['spring','summer','fall','winter']
                              ,inplace=True)

grouped_data
plt.figure(figsize=(12,8))


plt.pie(grouped_data['cnt'],
       labels=grouped_data['season'],
       autopct='%.1f')


plt.suptitle('Percentage count of Bike Rentals by Season');

plt.show()
# customize par chart adding white frame line around each section

plt.figure(figsize=(12,8))


plt.pie(grouped_data['cnt'],
       labels=grouped_data['season'],
       autopct='%.1f',
       wedgeprops={'linewidth':4,
                  'edgecolor' : "white"})


plt.suptitle('Percentage count of Bike Rentals by Season');

plt.show()
