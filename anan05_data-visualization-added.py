# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from folium.plugins import HeatMap

import folium

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data

data_offense_code = pd.read_csv("/kaggle/input/crimes-in-boston/offense_codes.csv",encoding="gbk")

data_crime = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv",encoding="gbk")
#remove nnull entries from lat and long

print(data_crime.isnull().sum())

data_crime = data_crime.dropna(subset=["Lat"])

data_crime.isnull().sum()
# Filtering offenses that are greater than 10000 in this time period

plot_greaterthan10k = data_crime['OFFENSE_CODE_GROUP'].value_counts()[data_crime['OFFENSE_CODE_GROUP'].value_counts() > 10000]

plot_greaterthan10k.plot(kind = 'bar' , title = 'Most common offenses 2015-2018', figsize=(20,5))

plt.xlabel("Offenses")

plt.ylabel("Count")

type(plot_greaterthan10k)

# function to get offense locations

def GetOffenseLocation(offenseName):

    offense = data_crime.loc[data_crime['OFFENSE_CODE_GROUP'] == offenseName][['Lat','Long']]

    offense = offense.values.tolist()

    return offense
#Display on google map the offenses wth color coding and legend 

colors = [

    'red',

    'blue',

    'gray',

    'darkred',

    'lightred',

    'orange',

    'beige',

    'green',

    'darkgreen',

    'lightgreen',

    'darkblue',

    'lightblue',

]

#counter to switch to next color

x=-1

#add a map

boston_shooting_map = folium.Map(location=[42.3601, -71.0589],zoom_start = 10)

for offense in plot_greaterthan10k.index:

    offense_location = GetOffenseLocation(offense)

    x = x+1

    for i in offense_location[:100]:  

        

        folium.Marker(\

             location=i, \

             popup=offense, \

             icon=folium.Icon(color = colors[x] , icon='glyphicon-remove')).add_to(boston_shooting_map)    

display(boston_shooting_map)

#Analysis of daily, monthly, weekly and yearly crimes

#daily from 2014 to 2018

plt.figure(figsize=(15,7)) 

sns.countplot(x=data_crime.DAY_OF_WEEK)
#monthly from 2014 to 2018

sizes_month = []    

labels = 'January', 'Febuary', 'March', 'April', 'May', 'Jun', 'July', 'August', 'September', 'October', 'November', 'December'

explode = (0, 0,0,0,0,0,0,0.2,0,0,0,0)

for i in range(12):

    i+=1

    sizes_month.append(len(data_crime[data_crime['MONTH']==i]))

plt.figure(figsize=(9,7))

plt.pie(sizes_month, autopct='%.1f%%', labels=labels, shadow=True, explode= explode )

plt.show()
#weekly

data_crime.index = pd.to_datetime( data_crime['OCCURRED_ON_DATE'])

weekly_crime = data_crime.MONTH.resample('W').count()

print(weekly_crime.index)

#sns.relplot(x=weekly_crime, y=weekly_crime.index, kind="line")

plt.figure(figsize=(13,7))

plt.plot(weekly_crime)

plt.xlabel('Weeks')

plt.ylabel('Count')

plt.title('Crimes per week')
#yearly analysis

yearly_analysis = data_crime.YEAR.value_counts()

print(type(yearly_analysis))

plt.figure(figsize=(15,16))

plt.subplot(2,2,1)

sns.barplot(x=yearly_analysis.index, y = yearly_analysis)

plt.figure(figsize=(15,16))

plt.subplot(2,2,2)

sns.lineplot(x=yearly_analysis.index, y = yearly_analysis)