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

import seaborn as sns

from pandas_profiling import ProfileReport
# Loading the Dataset

df = pd.read_csv("/kaggle/input/montcoalert/911.csv")
# Viewing the head of dataset. How the data looks.

df.head()
# pfr = ProfileReport(df, title="911_EDA")

# pfr
df.info()
df.size
df.shape
df.describe()
# Null Value check

df.isnull().sum()
#Since e has a constant value , dropping the column

df.drop(['e'],axis=1,inplace=True)
# Fill NaN values with zero

df.fillna(0 ,inplace=True)

df.isnull().sum()
df.head()
from datetime import datetime

import calendar
#We can convert a string to datetime using strptime() function

df['timeStamp'] = df['timeStamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
#Converting to Columns



df['Year'] = df['timeStamp'].apply(lambda t: t.year)

df['Month'] = df['timeStamp'].apply(lambda t: calendar.month_name[t.month])

df['Day'] = df['timeStamp'].apply(lambda t: t.day)

df['day name'] = df['timeStamp'].apply(lambda t: calendar.day_name[t.dayofweek])

df['Date'] = df['timeStamp'].apply(lambda t: t.date())
# Since we have different annotations for different time of a day, 

# So to get each part of day from Hours we define a function

def partofday(hour):

    return (

        "morning" if 5 <= hour <= 11

        else

        "afternoon" if 12 <= hour <= 17

        else

        "evening" if 18 <= hour <= 22

        else

        "night"

    )
# Getting the part of day column

df["part_of_day"] = df['timeStamp'].apply(lambda t: partofday(t.hour))
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

df['Reason_subtype'] = df['title'].apply(lambda title: title.split(':')[1])
#Since title column is not needed anymore,hence dropping the column

df.drop(['title'],axis=1,inplace=True)



#view the data head now after subsequent changes made above

df.head()
# Visualize Which part of day mostly the calls are made

sns.countplot(x='part_of_day',data=df)
print (df['part_of_day'].value_counts(normalize=True) * 100)

_, ax = plt.subplots()

ax.set_xlabel("Percentage")

ax.set_ylabel("part_of_day")

sns.set(rc={'figure.figsize': (9,8)})

(df['part_of_day'].value_counts(normalize=True) * 100).plot.barh()
sns.countplot(x='part_of_day',data=df, hue='Reason')
df['Month'].value_counts().plot(kind='bar',figsize=(12,12))
plt.figure(figsize=(15,8))

sns.countplot(data=df, x='Month', order=df.Month.value_counts().index, hue='Reason')
df['Year'].value_counts().plot(kind='bar',figsize=(10,10))
# View the locations in map using Folium map

import folium

from folium import plugins

from io import StringIO

import matplotlib.gridspec as gridspec



#Take any random Town lat and long,for Example 3rd Town from df.head

location = folium.Map([df['lat'][3],df['lng'][3]], zoom_start=15,tiles='OpenStreetMap') 

 



location
# We shall find out the maximum calls made for any reason (Top 15)



Calls_made = df['Reason_subtype'].value_counts()[:15]

plt.figure(figsize=(12, 8))

x = list(Calls_made.index)

y = list(Calls_made.values)

x.reverse()

y.reverse()



plt.title("Most emergency reasons of calls")

plt.ylabel("Reason")

plt.xlabel("Number of calls")



plt.barh(x, y)
Calls_per_daytiming = df.groupby(by=['Reason','part_of_day']).count()['Reason_subtype'].unstack().reset_index()

Calls_per_daytiming
dayHour = df.groupby(by=['day name','part_of_day']).count()['Reason_subtype'].unstack()

dayHour.head()
# Visualize through Heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(dayHour,cmap='mako')
plt.figure(figsize=(12, 8))

sns.clustermap(dayHour, cmap='vlag')
# Filtering the EMS calls and its Reason Types

filtered_EMS_calls = df[(df['Reason']!='Fire') & (df['Reason']!='Traffic')]['Reason_subtype']

filtered_EMS_calls
# Visualising the top 30 Reason among EMS category

print (filtered_EMS_calls.value_counts(normalize=True) * 100)

_, ax = plt.subplots()

ax.set_xlabel("Percentage")

ax.set_ylabel("Reason Subtypes Under EMS")

sns.set(rc={'figure.figsize': (9,8)})

(filtered_EMS_calls.value_counts(normalize=True) * 100)[:30].plot(kind='barh',figsize=(10,10))
#Filtering Traffic calls and its sub categories

filtered_traffic = df[(df['Reason']!='EMS') & (df['Reason']!='Fire')]['Reason_subtype']

filtered_traffic
# Visualising the top reasons among Traffic category

print (filtered_traffic.value_counts(normalize=True) * 100)

_, ax = plt.subplots()

ax.set_xlabel("Percentage")

ax.set_ylabel("Reason Subtypes Under Traffic")

sns.set(rc={'figure.figsize': (9,8)})

(filtered_traffic.value_counts(normalize=True) * 100)[:30].plot(kind='barh',figsize=(10,10))
plt.figure(figsize=(18,7))

df[df['Reason']=='EMS'].groupby('day name').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
plt.figure(figsize=(18,7))

df[df['Reason']=='EMS'].groupby('part_of_day').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
plt.figure(figsize=(18,7))

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
plt.figure(figsize=(18,7))

df.groupby('Date').count()['twp'].plot()

plt.tight_layout()


plt.figure(figsize=(18,7))

df[df['Reason']=='EMS'].groupby(by=['Reason','part_of_day']).count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
plt.figure(figsize=(18,7))

df[df['Reason']=='EMS'].groupby(by=['Reason_subtype','part_of_day']).count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
dayMonth = df.groupby(by=['day name','Month']).count()['Reason'].unstack()

dayMonth.head()
#Now create a HeatMap using this new DataFrame.

plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='PuBuGn',linewidths=2)
# Finding the Unique Townships in the dataset

df[["twp","Reason"]].groupby('twp').count().sort_values("Reason")



Unique_Town = df.twp.unique()

townarray = []

corresponding_reason = []

for town in Unique_Town:

        town_len =  len(df[df["twp"] == town])    

        townarray.append(town_len)
len(Unique_Town)
# Let us see the visualization for top 30 Towns in a sorted manner (Higer to Lower 911 calls)

tdf = pd.DataFrame(

    {'Town': Unique_Town,

     'Count' : townarray,

     })



tdf_sort = tdf.sort_values("Count", ascending=False).reset_index(drop=True)



f, ax = plt.subplots(figsize=(10,8)) 

ax.set_yticklabels(tdf.Town, rotation='horizontal', fontsize='large')

g = sns.barplot(y = tdf_sort.Town[:30], x= tdf_sort.Count)



plt.show()

# Let us see the visualization for top 30 Towns in a sorted manner (Lower to Higher 911 Calls)

tdf_sort = tdf.sort_values("Count", ascending=True).reset_index(drop=True)



f, ax = plt.subplots(figsize=(10,8)) 

ax.set_yticklabels(tdf.Town, rotation='horizontal', fontsize='large')

g = sns.barplot(y = tdf_sort.Town[:30], x= tdf_sort.Count)

# g = sns.barplot(y = tdf_sort.Town[:-15], x= tdf_sort.Count)

plt.show()

# Finding out the EMS reason of call made for the unique towns

for town in Unique_Town:

    EMS = df[(df['Reason']!='Fire') & (df['Reason']!='Traffic')][["twp","Reason"]]
print (EMS.twp.value_counts()[0:15])

plt.figure(figsize=(18,7))

EMS.twp.value_counts().plot.bar()
# Finding out the Traffic reason of call made for the unique towns

for town in Unique_Town:

    Traffic = df[(df['Reason']!='Fire') & (df['Reason']!='EMS')][["twp","Reason"]]



    

print (Traffic.twp.value_counts()[0:15])

plt.figure(figsize=(18,7))

Traffic.twp.value_counts().plot.bar()
# Finding out the Fire reason of call made for the unique towns

for town in Unique_Town:

    Fire = df[(df['Reason']!='Traffic') & (df['Reason']!='EMS')][["twp","Reason"]]



    

print (Fire.twp.value_counts()[0:15])

plt.figure(figsize=(18,7))

Fire.twp.value_counts().plot.bar()
sns.catplot(x="day name", hue="Reason",



                data=df, kind="count",



                height=12, aspect=1);


def color(count): 

    if count in range(0,5000): 

        col = 'green'

    elif count in range(5001,10000): 

        col = 'blue'

    elif count in range(10001,20000): 

        col = 'orange'

    else: 

        col='red'

    return col 
#Back to Folium Map, let us view the Towns and Area in map from where most no of calls are made.

for lat,lng,twp,area,count in zip(df['lat'], df['lng'],df['twp'],df['addr'],tdf_sort['Count']):

    folium.Marker(location=[lat,lng],

                        popup = ('Town: ' + str(twp).capitalize() + '<br>' 'Area: ' +str(area)

                                + '<br>' 'Calls Made: ' + str(count)),

                        icon= folium.Icon(color=color(count),  

                        icon_color='yellow', icon = 'info-sign')

                        ).add_to(location)

    

location



#Zoom Out to view the different areas with different colour marks as per no of calls