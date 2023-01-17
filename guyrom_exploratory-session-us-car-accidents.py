# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import datetime
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_May19.csv')

df.columns
df.info()
print('There are {} accidents in the data'.format(len(df)))
df.Severity.value_counts(normalize=True).sort_index().plot.bar()

plt.grid()

plt.title('Severity')

plt.xlabel('Severity')

plt.ylabel('Fraction');
bool_cols = [col for col in df.columns if df[col].dtype ==np.dtype('bool')]

booldf = df[bool_cols]

not_one_hot = booldf[booldf.sum(axis=1) > 1]

print('There are {} non one hot metadata rows, which are {:.1f}% of the data'.format(len(not_one_hot),100*len(not_one_hot)/len(df)))
bools = booldf.sum(axis=0)
bools
bools.plot.pie(figsize=(13,13))

plt.ylabel('')

plt.title('Proximity to Traffic Object');
st = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')

end = pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')
diff = (end-st)

top20 = diff.astype('timedelta64[m]').value_counts().nlargest(20)

print('top 20 accident durations correspond to {:.1f}% of the data'.format(top20.sum()*100/len(diff)))

(top20/top20.sum()).plot.bar(figsize=(14,14))

plt.title('Accident Duration [Minutes]')

plt.xlabel('Duration [minutes]')

plt.ylabel('Fraction');
df.head()
df['time'] = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')

df = df.set_index('time')

df.head()
freq_text = {'D':'Daily','W':'Weekly','Y':'Yearly'}

plt.subplots(1,3,figsize=(21,7))

for i, (fr,text) in enumerate(freq_text.items(),1):

    plt.subplot(1,3,i)

    sample = df.ID['2016':].resample(fr).count()

    sample.plot(style='.')

    plt.title('Accidents, {} count'.format(text))

    plt.xlabel('Date')

    plt.ylabel('Accident Count');
years = ['2016','2017','2018']

fig, _ = plt.subplots(1,3,figsize=(21,7), sharex='all', sharey='all')



fig.suptitle('Acccidents by month for Different Years')

plt.xlabel('month')

plt.ylabel('Accidents')

for i, year in enumerate(years,1):

    plt.subplot(1,3,i)

    sample = df.loc[year].ID.resample('M').count()

    sample.plot()

    plt.ylim(0,100000)

    plt.title('Accidents, {} count'.format(text))

    plt.xlabel('Date')

    plt.ylabel('Accident Count');
df['Weekday'] = df.index.weekday_name

weekday = df.groupby('Weekday').ID.count()

weekday = weekday/weekday.sum()

dayOfWeek=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

weekday[dayOfWeek].plot.bar()

plt.title('Acccidents by Weekday')

plt.xlabel('Weekday')

plt.ylabel('Accidents');
years = ['2016','2017','2018']

fig, _ = plt.subplots(1,3,figsize=(21,7), sharex='all', sharey='all')



fig.suptitle('Acccidents by Weekday for Different Years')

plt.xlabel('Weekday')

plt.ylabel('Accidents')

for i, year in enumerate(years,1):

    weekday = df.loc[year].groupby('Weekday').ID.count()

    weekday = weekday/weekday.sum()

    dayOfWeek=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    plt.subplot(1,3,i)

    plt.title(year)

    weekday[dayOfWeek].plot.bar()
import folium

sdf = df.sample(10000)

m = folium.Map(location=[sdf.Start_Lat.mean(),sdf.Start_Lng.mean()], zoom_start=4,tiles="CartoDB dark_matter")

sdf.apply(lambda row:folium.CircleMarker(location=[row["Start_Lat"], row["Start_Lng"]],radius=1).add_to(m), axis=1)

m.save('./map.html')
from IPython.display import FileLink

print("The following is a map of 10000 accidents")

FileLink(r'map.html')