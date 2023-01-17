import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
%matplotlib inline

plt.rcParams['figure.figsize'] = [20.0, 7.0]
plt.rcParams.update({'font.size': 22})

sns.set_style('whitegrid')
sns.set_context('talk')
df = pd.read_csv('../input/bike_sharing_daily.csv')
#view column names
df.columns
df.info()
#view summary stats of numeric variables
df.describe()
df.corr()
#rename columns
df = df.rename(columns={'dteday':'datetime',
                        'yr':'year',
                        'mnth':'month',
                        'weathersit':'weather',
                        'hum':'humidity',
                        'cnt':'total_rides'})

#set categorical variables
##why set as categories??
df['season'] = df['season'].astype('category')
df['year'] = df['year'].astype('category')
df['month'] = df['month'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weather'] = df['weather'].astype('category')
df.head()
print('Winter vs Spring')
print(ttest_ind(df.total_rides[df['season'] == 1], df.total_rides[df['season'] == 2]))
print('Winter vs Summer')
print(ttest_ind(df.total_rides[df['season'] == 1], df.total_rides[df['season'] == 3]))
print('Winter vs Fall')
print(ttest_ind(df.total_rides[df['season'] == 1], df.total_rides[df['season'] == 4]))
print('Spring vs Fall')
print(ttest_ind(df.total_rides[df['season'] == 2], df.total_rides[df['season'] == 4]))
print('Spring vs Summer')
print(ttest_ind(df.total_rides[df['season'] == 2], df.total_rides[df['season'] == 3]))
print('Summer vs Fall')
print(ttest_ind(df.total_rides[df['season'] == 3], df.total_rides[df['season'] == 4]))
fig, ax = plt.subplots()
sns.barplot(data=df[['season','total_rides']],
            x='season',
            y='total_rides',
            ax=ax)

plt.title('Capital Bikeshare Ridership by Season')
plt.ylabel('Total Rides')
plt.xlabel('Season')

tick_val=[0, 1, 2, 3]
tick_lab=['Winter', 'Spring', 'Summer', 'Fall']
plt.xticks(tick_val, tick_lab)

plt.show()
fig, ax = plt.subplots()
sns.barplot(data=df[['month','total_rides']], x='month', y='total_rides', ax=ax)

plt.title('Capital Bikeshare Ridership by Month')
plt.ylabel('Total Rides')
plt.xlabel('Month')

tick_val=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
tick_lab=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(tick_val, tick_lab)

plt.show()
df['day_of_month'] = df.datetime.str[-2:]
df.head()

fig, ax = plt.subplots()
sns.pointplot(data=df[['day_of_month', 'total_rides', 'season']],
              x='day_of_month',
              y='total_rides',
              hue='season',
              ax=ax)

plt.title('Capital Bikeshare Ridership by Day')
plt.ylabel('Total Rides')
plt.xlabel('Day of Month')

leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['Winter', 'Spring', 'Summer', 'Fall'], title='Season', bbox_to_anchor=(1, 1), loc=2)

plt.show()
ttest_ind(df['registered'], df['casual'])
fig = plt.subplot()
sns.boxplot(data=df[['total_rides', 'casual', 'registered']])
fig, ax = plt.subplots()
sns.pointplot(data=df[['month', 'casual', 'registered']],
              x='month',
              y='casual',
              ax=ax,
              color='orange')

sns.pointplot(data=df[['month', 'casual', 'registered']],
              x='month',
              y='registered',
              ax=ax,
              color='green')

tick_val=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
tick_lab=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(tick_val, tick_lab)

plt.title('Casual and Registered Bikeshare Ridership by Month')
plt.ylabel('Total Rides')
plt.xlabel('Month')

plt.show()
plt.rcParams['figure.figsize'] = [10.0, 10.0]
sns.set_context('talk', font_scale=0.8)

g = sns.FacetGrid(data=df,
               col='season',
               row='weather',hue='season')
g.map(plt.hist,'total_rides')

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Capital Bikeshare Ridership by Weather Type')

g.set_xlabels('Total Rides')
g.set_ylabels('Frequency')

plt.show()
