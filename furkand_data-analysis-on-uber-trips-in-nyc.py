%pylab inline
import pandas as pd
import seaborn as sns
uber_data = pd.read_csv('../input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv')
uber_data.head()
uber_data['Date/Time']=uber_data['Date/Time'].map(pd.to_datetime)
uber_data.head()
uber_data.info()
uber_data['Day'] = uber_data['Date/Time'].apply(lambda x: x.day)
uber_data['WeekDay'] = uber_data['Date/Time'].apply(lambda x: x.weekday())
uber_data['hour'] = uber_data['Date/Time'].apply(lambda x: x.hour)
uber_data.tail()

plt.figure(figsize=(10,6))
uber_data['Day'].hist(bins=30,rwidth=0.9,range=(0.5,30.5))
plt.xlabel('Day of Month')
plt.ylabel('frequency')
plt.title('Uber - Daily Frequency - Aug 2014')
for x,rows in uber_data.groupby('Day'):
    print((x,len(rows)))
## This is not so useful so we will write a function instead. A simple lambda func will do.
by_date = uber_data.groupby('Day').apply(lambda x: len(x))
by_date
plt.figure(figsize=(10,6))
by_date.plot()
by_date_sorted= by_date.sort_values()
by_date_sorted
plt.figure(figsize=(10,6))
bar(range(0,31),by_date_sorted)
plt.xlabel('Day of Month')
plt.ylabel('frequency')
plt.title('Uber - Daily Frequency - Aug 2014')
xticks(range(1,31),by_date_sorted.index)
plt.figure(figsize=(10,6))

uber_data.hour.hist(bins=24, range=(0,25))
by_hour = uber_data.groupby('hour').apply(lambda x: len(x))
by_hour_sorted = by_hour.sort_values()


plt.figure(figsize=(10,6))
bar(range(0,24),by_hour_sorted)
plt.xlabel('Hour of Day')
plt.ylabel('Frequency')
plt.title('Uber - Hourly - Thru Aug 2014')
xticks(range(0,24),by_hour_sorted.index)

plt.figure(figsize=(10,6))

hist(uber_data.WeekDay, bins=7,range=(-0.5,6.5),rwidth=0.8)
xticks(range(7),['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

by_hour_week = uber_data.groupby(['WeekDay','hour']).apply(lambda x: len(x))

by_hour_weeek = by_hour_week.unstack()



by_hour_weeek
plt.figure(figsize=(10,6))

cmap = sns.cm.rocket_r

sns.heatmap(by_hour_weeek, annot=False, cmap=cmap)
print(uber_data['Lon'].max())
print(uber_data['Lon'].min())
print(uber_data['Lat'].max())
print(uber_data['Lat'].min())
plt.figure(figsize=(25,15))


plot(uber_data['Lon'], uber_data['Lat'], '.', ms=0.5)
xlim(-74.2, -73.7)
ylim(40.7,41)

