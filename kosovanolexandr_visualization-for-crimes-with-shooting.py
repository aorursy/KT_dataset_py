import numpy as np
import pandas as pd
# visualization

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
def bar_chart(list1, list2):
    objects = list1
    y_pos = np.arange(len(objects))
    performance = list2
 
    plt.figure(figsize=(20,10))    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number') 
    plt.show()
    
    return 0
df = pd.read_csv('../input/data.csv')
df = df.loc[df.SHOOTING == 1]
df.shape
df = df[['DAY_OF_WEEK', 'DISTRICT', 'HOUR', 'Lat',
       'Long', 'MONTH', 'OCCURRED_ON_DATE', 'OFFENSE_CODE_GROUP',
       'REPORTING_AREA', 'SHOOTING', 'UCR_PART', 'YEAR', 'Day', 'Night',
       'ToNight', 'ToDay', 'DAY', 'temperatureMin', 'temperatureMax',
       'precipitation', 'snow', 'temperatureDifference']]
day_of_week = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
i = 0
day_number = list()

while i < 7:
    day_number.append(len(df.loc[df['DAY_OF_WEEK'] == day_of_week[i]]))
    
    i +=1
bar_chart(day_of_week, day_number)
plt.figure(figsize=(20,10))
df['DISTRICT'].value_counts().plot.bar()
plt.show()
i = 0
hour_number = list()

while i < 24:
    hour_number.append(len(df.loc[df['HOUR'] == i]))
    i +=1
bar_chart(list(range(0,24)), hour_number)
i = 1
list_month = list()

while i <= 12:
    list_month.append(len(df.loc[df['MONTH'] == i]))
    i+=1
bar_chart(list(range(1,13)), list_month)
location = df[['Lat','Long']]
location = location.dropna()

location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]
x = location['Long']
y = location['Lat']
# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”

sns.jointplot(x, y, kind='scatter')
sns.jointplot(x, y, kind='hex')
sns.jointplot(x, y, kind='kde')
plt.figure(figsize=(20,10))
df.YEAR.value_counts().plot.bar()
plt.show()
plt.figure(figsize=(20,10))
df.Day.value_counts().plot.bar()
plt.show()
plt.figure(figsize=(20,10))
sns.barplot(x=df.Day.value_counts().index, y=df.Day.value_counts())
plt.figure(figsize=(20,10))
df.ToNight.value_counts().plot.bar()
plt.show()
plt.figure(figsize=(20,10))
df.ToDay.value_counts().plot.bar()
plt.show()
# min           0.000000

df.temperatureMin.describe()
# min           0.000000

df.temperatureMax.describe()
df.precipitation.describe()
df.snow.describe()
df.temperatureDifference.describe()
