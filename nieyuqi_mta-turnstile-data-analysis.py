import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.ticker import FuncFormatter
matplotlib.style.use('ggplot')
import sqlite3 
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

mta = pd.read_csv('../input/mta-turnstile-data-first-half-of-2018/turnstile_18_1.csv', sep = ',',header = None)
mta.columns =['C/A','unit','SCP','Station','date','time','desc','entries','exits']      
mta.head()
mta['datetime'] = pd.to_datetime(mta.date + ' ' + mta.time, format='%m/%d/%Y  %H:%M:%S')
mta['turnstile'] = mta['C/A'] + '-' + mta['unit'] + '-' + mta['SCP']

mta = mta[(mta.datetime >= '01-01-18 00:00:00') & 
          (mta.datetime <'07-01-18 00:00:00')]
mta.head()
print('Descriptions of entries:')
print(mta['entries'].describe())
print('')
print('Descriptions of exits:')
print(mta['exits'].describe())
# group data by turnstile, sort each turnstile by datetime
# Create new columns en_diff and ex_diff for each unique turnstile
# turn cumulative counts into counts per interval

mta_sorted = mta.sort_values(['turnstile', 'datetime'])
mta_sorted = mta_sorted.reset_index(drop = True)

turnstile_grouped = mta_sorted.groupby(['turnstile'])

mta_sorted['entries_diff'] = turnstile_grouped['entries'].transform(pd.Series.diff)
mta_sorted['exits_diff'] = turnstile_grouped['exits'].transform(pd.Series.diff)

mta_sorted.head()
del mta
# check distribution of entries_diff and exits_diff
print('Descriptions of entries_diff:')
print(mta_sorted['entries_diff'].describe())
print('')
print('Descriptions of exits_diff:')
print(mta_sorted['exits_diff'].describe())
print('Number of negative entries_diff: %d' %len(mta_sorted['entries_diff'][mta_sorted['entries_diff'] < 0]))
print('Number of negative exits_diff: %d' %len(mta_sorted['exits_diff'][mta_sorted['exits_diff'] < 0]))
print('Number of unqiue turnstiles: %d' %len(mta_sorted['turnstile'].unique()))
print('Number of NaN rows: %d' %len(mta_sorted[mta_sorted['entries_diff'].isnull()]))
mta_sorted['entries_diff'] = mta_sorted['entries_diff'].fillna(0)
mta_sorted['exits_diff'] = mta_sorted['exits_diff'].fillna(0)

mta_sorted['entries_diff'][mta_sorted['entries_diff'] < 0] = 0 
mta_sorted['exits_diff'][mta_sorted['exits_diff'] < 0] = 0 

mta_sorted['entries_diff'][mta_sorted['entries_diff'] >= 6000] = 0 
mta_sorted['exits_diff'][mta_sorted['exits_diff'] >= 6000] = 0
mta_h1 = mta_sorted[['turnstile','Station', 'datetime','date','time', 'entries_diff','exits_diff']]
mta_h1['busy'] = mta_h1['entries_diff'].values + mta_h1['exits_diff'].values

del [mta_sorted]
mta_h1.shape
mta_h1.head()
top_10 =mta_h1.groupby(['Station']).agg({'busy': sum}).sort_values(by = 'busy', ascending = False).head(10)

fig, ax = plt.subplots(figsize=(25, 8))
top_10.sort_values(by = 'busy',ascending=True).plot(kind='barh', color ='orange',ax=ax)
ax.set(title='Top 10 Stations by Total Entries and Exits (January-June 2018)', xlabel='total traffic', ylabel='')
ax.legend().set_visible(False)
mta_penn = mta_h1[mta_h1['Station'] == '34 ST-PENN STA']
print('Number of turnstiles at 34-PENN STATION: %d' % len(mta_penn.turnstile.unique()))
penn_turnstile =mta_penn.groupby(['turnstile']).agg({'busy': sum}).sort_values(by = 'busy', ascending = False)

fig, ax = plt.subplots(figsize=(25, 8))
penn_turnstile.sort_values(by = 'busy',ascending=True).plot(kind='barh', color ='steelblue',ax=ax)
ax.set(title='Total Traffic by Turnstile (34 ST-PENN STA)', xlabel='total traffic', ylabel='')
ax.legend().set_visible(False)
top_5 =mta_penn.groupby(['turnstile']).agg({'busy': sum}).sort_values(by = 'busy', ascending = False).head()

fig, ax = plt.subplots(figsize=(25, 8))
top_5.sort_values(by = 'busy',ascending=True).plot(kind='barh', color ='steelblue',ax=ax)
ax.set(title='Top 5 Turnstiles by Total Traffic (34 ST-PENN STA)', xlabel='total traffic', ylabel='')
ax.legend().set_visible(False)
bottom_5 =mta_penn.groupby(['turnstile']).agg({'busy': sum}).sort_values(by = 'busy', ascending = True).head()

fig, ax = plt.subplots(figsize=(25, 8))
bottom_5.sort_values(by = 'busy',ascending=True).plot(kind='barh', color ='steelblue',ax=ax)
ax.set(title='Bottom 5 Turnstiles by Total Traffic (34 ST-PENN STA)', xlabel='total traffic', ylabel='')
ax.legend().set_visible(False)
mta_penn = mta_h1[(mta_h1['Station'] == '34 ST-PENN STA')]
mta_penn['time'] = pd.to_datetime(mta_penn['time'], format = '%H:%M:%S') # have to be datetime format so that we could resample
mta_penn_grouped = mta_penn.groupby(['time']).agg({'busy': sum}).sort_values(by = 'busy', ascending = False)
mta_penn_grouped.resample('60T',convention='end').sum().sort_values(by = 'busy', ascending = False).head()
mta_h1busy = mta_h1[['date', 'busy']].groupby(['date']).sum().reset_index()
mta_h1busy['date'] = pd.to_datetime(mta_h1busy['date'])
mta_h1busy.set_index('date')['busy'].plot(color = 'steelblue')
plt.title('Daily Total Traffic for First Half of 2018') 
plt.show()
del [mta_h1busy]
mta_h1['weekday'] = (mta_h1['datetime']).dt.weekday_name
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
mta_h1['weekday'] = pd.Categorical(mta_h1['weekday'], categories=weekdays, ordered = True)
weekday_ave = mta_h1[['entries_diff', 'exits_diff', 'busy', 'weekday']].groupby('weekday').sum().reset_index().sort_values(by = 'weekday')
weekday_ave['emgergency'] = (weekday_ave['entries_diff'] - weekday_ave['exits_diff'])/weekday_ave['busy']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

p1 = weekday_ave['busy'].plot(kind='bar', color='steelblue', ax=ax1, label = 'entries')
p1.legend(loc = 1)
p2 = weekday_ave['exits_diff'].plot(kind='bar', color='orange', ax=ax1, label = 'exits')
p2.legend(loc = 1)
p3 = weekday_ave['emgergency'].plot(kind='line', dashes = [5, 2], color='maroon', ax=ax2, label = 'EE rate')
p3.legend(loc = (0.775, 0.75)) 

plt.xticks((0,1,2,3,4,5,6),('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
ax1.set_xlabel('week of day')
ax1.set_ylabel('total traffic', color='steelblue')

ax2.set_ylabel('emgergency exit rate', color='maroon')
ax2.grid(False)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y))) 
plt.title('Traffic & Emgergency Exits by Week of Day', size = 12) 
plt.show()
mta_penn = mta_h1[mta_h1['Station'] == '34 ST-PENN STA'].groupby(by = 'date').agg({'busy' : sum}).reset_index()
mta_penn['date'] = pd.to_datetime(mta_penn['date'])
mta_penn['month'] = mta_penn['date'].dt.month
penn_month = mta_penn[['month', 'busy']].groupby('month')
penn_mean = penn_month.mean()
mta_penn['month']=mta_penn['month'].apply(str)
p1 = penn_mean.plot(kind='line', color = 'orange', dashes = (6, 2)) 
p1.legend(['mean'])
plt.boxplot([mta_penn[mta_penn['month'] == '1']['busy'].values,mta_penn[mta_penn['month'] == '2']['busy'].values,mta_penn[mta_penn['month'] == '3']['busy'].values,mta_penn[mta_penn['month'] == '4']['busy'].values,mta_penn[mta_penn['month'] == '5']['busy'].values,mta_penn[mta_penn['month'] == '6']['busy'].values])
plt.xticks((1,2,3,4,5,6),('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'))
plt.title('Daily Traffic by Month (34 ST-PENN STATION, quantiles & mean)', size = 13)
plt.show()
station_of_interest = '34 ST-PENN STA'
mta_penn = mta_h1[mta_h1['Station'] == station_of_interest]
time_interval = '4H'

mta_penn.head()
stations_day_time = mta_penn[['Station', 'datetime', 'entries_diff','exits_diff','busy']]
stations_day_time_group = stations_day_time.groupby(['Station','datetime'], as_index=False)
stations_day_time = stations_day_time_group[['entries_diff', 'exits_diff','busy']].sum()
rounded_day_time = stations_day_time.set_index('datetime').groupby(['Station'])
rounded_day_time = rounded_day_time.resample(time_interval, convention='end').sum()

print ('Station of interest: ' + station_of_interest)
print ('Sample size before resampling:')
print (len(stations_day_time[stations_day_time.Station == station_of_interest]))
print ('Sample size after resampling:')
print (len(rounded_day_time.loc[station_of_interest]))
rounded_day_time = rounded_day_time.reset_index()
rounded_day_time.head()
stations_day_time = rounded_day_time
stations_day_time['DAY'] = stations_day_time['datetime'].dt.dayofweek
stations_day_time['TIME'] = stations_day_time['datetime'].dt.time
stations_day_time_group = stations_day_time.groupby(['Station','DAY','TIME'])
stations_day_time_group = stations_day_time_group['entries_diff'].mean().reset_index()
draw_station = stations_day_time_group

station_heatmap = draw_station.set_index(['DAY', 'TIME']).entries_diff.unstack(0)
weekdays = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
station_heatmap.columns = [weekdays[c] for c in station_heatmap.columns]

fig, ax = plt.subplots(figsize=(12, 8))
ax.set(title='Entries: ' + station_of_interest , xlabel='', ylabel='Time')
sns.heatmap(station_heatmap,ax=ax, cmap='Blues')
plt.show()
stations_day_time_group = stations_day_time.groupby(['Station','DAY','TIME'])
stations_day_time_group = stations_day_time_group['exits_diff'].mean().reset_index()
draw_station = stations_day_time_group

station_heatmap = draw_station.set_index(['DAY', 'TIME']).exits_diff.unstack(0)
weekdays = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
station_heatmap.columns = [weekdays[c] for c in station_heatmap.columns]

fig, ax = plt.subplots(figsize=(12, 8))
ax.set(title='Exits: ' + station_of_interest , xlabel='', ylabel='Time')
sns.heatmap(station_heatmap,ax=ax, cmap='Blues')
plt.show()
stations_day_time_group = stations_day_time.groupby(['Station','DAY','TIME'])
stations_day_time_group = stations_day_time_group['busy'].mean().reset_index()
draw_station = stations_day_time_group

station_heatmap = draw_station.set_index(['DAY', 'TIME']).busy.unstack(0)
weekdays = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
station_heatmap.columns = [weekdays[c] for c in station_heatmap.columns]

fig, ax = plt.subplots(figsize=(12, 8))
ax.set(title='Traffic: ' + station_of_interest , xlabel='', ylabel='Time')
sns.heatmap(station_heatmap,ax=ax, cmap='Blues')
plt.show()
conn = sqlite3.connect('Station.db')
col_names = ['unit','C/A','Station','LINENAME','DIVISION']
remote = pd.read_excel('../input/mta-remoteboothstations/Remote-Booth-Station.xls', names=col_names).drop_duplicates(['unit','C/A'])
remote.head()
remote.to_sql(name='remote_booth_station', con=conn, if_exists='replace')
cursor = conn.cursor()
cursor.execute("""SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'remote_booth_station'""")
desc = cursor.fetchall()
print(desc)
cursor = conn.cursor()
cursor.execute("""SELECT Station, COUNT(DISTINCT(unit)) FROM remote_booth_station GROUP BY Station ORDER BY COUNT(DISTINCT unit) DESC LIMIT 5""")
station = cursor.fetchall()
print(station)
cursor = conn.cursor()
cursor.execute("""SELECT COUNT(DISTINCT(Station)) FROM remote_booth_station """)
station = cursor.fetchall()
print(station)
cursor = conn.cursor()
cursor.execute("""SELECT COUNT(DISTINCT(DIVISION)) FROM remote_booth_station """)
station = cursor.fetchall()
print(station)
cursor = conn.cursor()
cursor.execute("""SELECT DIVISION, COUNT(DIVISION) FROM remote_booth_station GROUP BY DIVISION ORDER BY COUNT(DIVISION) DESC LIMIT 4""")
station = cursor.fetchall()
print(station)
cursor = conn.cursor()
cursor.execute("""SELECT LENGTH(LINENAME)as ilen, Station FROM remote_booth_station GROUP BY STATION ORDER BY ilen DESC LIMIT 10""")
station = cursor.fetchall()
print(station)
cursor.close()
conn.close()