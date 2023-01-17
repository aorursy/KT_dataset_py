#importing necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data loading
data=pd.read_csv('../input/austin_bikeshare_trips.csv')
data.head()

data.shape
#checking for missing data
data.isnull().sum()
start_station=data['start_station_name'].value_counts();

start_station.shape 
start_station
plt.figure(figsize=(40,40))
start_station.plot.barh()
#plt.show()
plt.savefig('demand.png')
import numpy as np
import seaborn as sns
duration=data['duration_minutes']
plt.figure(figsize=(10,10))
#sns.distplot(duration)

duration.plot(kind='hist')

#plt.xlim((duration.min(),duration.max()))
#plt.xticks(np.arange(duration.min(),duration.max(),5))


np.unique(data['year'])
year=data['year'].value_counts()
year.sort_index(inplace=True)
plt.figure()
year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Ride Count')
cot=data['checkout_time']
cot.dtype
cot.describe()
type(cot)
data['checkout_time']=pd.to_datetime(data['checkout_time'])
data['checkout_time']=[time.time() for time in data['checkout_time']]
data['Hour']=data['checkout_time'].apply(lambda x:x.hour)
print(data.head())
print(data['Hour'].unique())
print(data['Hour'].value_counts())
checkout=data['Hour'].value_counts()
plt.figure()
checkout.plot(kind='bar')
plt.xlabel('Hour')
plt.ylabel('Ride Count')
plt.show()

data.head(10)
import calendar
month=data['month'].value_counts()
fig, ax = plt.subplots()
# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
fig.canvas.draw()
month.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Ride Count')
mn=[calendar.month_name[int(x)] for x in month.index.values.tolist()]
ax.set_xticklabels(mn)
#ax.set_xticklabels(mn, rotation='vertical', fontsize=18)
plt.show()

bikeid=data['bikeid'].value_counts()
print(bikeid.head())
top_ten_bikeid=bikeid.head(10)
top_ten_bikeid.plot(kind='bar')
