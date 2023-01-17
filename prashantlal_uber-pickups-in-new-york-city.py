# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading the dataset of month of April and year 2014

uber_april_2014 = pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv')
#Glimpse of the data 

uber_april_2014.head()
#Checking for any missing value 

uber_april_2014.isnull().sum()
#More info about data

uber_april_2014.info()

uber_april_2014.shape

#After going through the data it's seem that the Date/Time column has string value

#Total 564516 observation 
#Converting string to datetime format

uber_april_2014['Date/Time']=uber_april_2014['Date/Time'].apply(lambda x : datetime.strptime(x,'%m/%d/%Y %H:%M:%S'))

uber_april_2014.info()
#After converting to date format, time specific column should be made but before that..

#checking for info in seconds and it's seem that all values are 0 so seconds column is not required

uber_april_2014['Date/Time'].apply(lambda x : x.second)
#columns contaning time specific info except second column

uber_april_2014['Year']=uber_april_2014['Date/Time'].apply(lambda x : x.year)

uber_april_2014['Month']=uber_april_2014['Date/Time'].apply(lambda x : x.month)



d={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thusday',4:'Friday',5:'Saturday',6:'Sunday'}

uber_april_2014['Day_of_week']=uber_april_2014['Date/Time'].apply(lambda x : x.weekday())

uber_april_2014['Day_of_week']=uber_april_2014['Day_of_week'].map(d)



uber_april_2014['Hour']=uber_april_2014['Date/Time'].apply(lambda x : x.hour)

uber_april_2014['Minute']=uber_april_2014['Date/Time'].apply(lambda x : x.minute)
uber_april_2014['Base'].value_counts()
#uber bases

d=dict(B02682='Schmecken',B02598='Hinter',B02617='Weiter',B02512='Unter',B02764='Danach_NY')

uber_april_2014['Base']=uber_april_2014['Base'].map(d)

uber_april_2014.head()
#Rearranging the column

uber_april_2014.columns.values

uber_april_2014=uber_april_2014[['Base', 'Year', 'Month', 'Day_of_week', 'Hour','Minute','Lat','Lon']]
#checking if everything went accordingly 

uber_april_2014.nunique()

#Everything is well fine, the only last thing left is to check base column
uber_april_2014.describe()

#observing the lat and lon column, the area of interest in very narrow 
#scatter ploting between lat and log

plt.figure(figsize=(15,8))

plt.scatter(uber_april_2014['Lat'],uber_april_2014['Lon'])

plt.show()
#to cluster the graph based on Base 

d=dict(Schmecken='0',Hinter='1',Weiter='2',Unter='3',Danach_NY='4')

uber_april_2014['BaseC']=uber_april_2014['Base'].map(d)

plt.figure(figsize=(18,10))

plt.scatter(uber_april_2014['Lat'],uber_april_2014['Lon'],c=uber_april_2014['BaseC'].astype(int),cmap='viridis')

plt.show()

uber_april_2014.drop('BaseC',axis=1,inplace=True)
#uber pickup frequency based on Hours of the day

plt.figure(figsize=(15,8))

sns.distplot(uber_april_2014['Hour'])

plt.show()

#uber is mostly busy around 3pm to 9pm and slightly busy in the morning around 6am to 10am
#uber pickup frequency based on days of the week

plt.figure(figsize=(15,8))

sns.countplot(uber_april_2014['Day_of_week'])

plt.show()

#people are choosing uber for weekdays more often then in weekends
# Insight: Unter Base traffic monitoring on the bases of Day_of_week and Hours 

plt.figure(figsize=(16,8))

sns.countplot(uber_april_2014[uber_april_2014['Base']=='Unter']['Day_of_week'],

              hue=uber_april_2014[uber_april_2014['Base']=='Unter']['Hour'],palette='seismic')

plt.title('Unter Base',size=30)

plt.legend(loc=(1.05,0))

plt.show()

#it's seem that uber has more pickup on Tuesday wednesday and thusday 

#people are opting for uber more in the evening(5pm to 9pm) period than the morning(6am to 9am) and the patter is same throughout the week
# Insight: Hinter Base traffic monitoring on the bases of Day_of_week and Hours 

plt.figure(figsize=(16,8))

sns.countplot(uber_april_2014[uber_april_2014['Base']=='Hinter']['Day_of_week'],

              hue=uber_april_2014[uber_april_2014['Base']=='Hinter']['Hour'],palette='seismic')

plt.title('Hinter Base',size=30)

plt.legend(loc=(1.05,0))

plt.show()

#it's seem that uber has less pickup on sunday and monday 

#the trend of pickup hour is as similar as observed in Unter traffic analysis
# Insight: Weiter Base traffic monitoring on the bases of Day_of_week and Hours 

plt.figure(figsize=(16,8))

sns.countplot(uber_april_2014[uber_april_2014['Base']=='Weiter']['Day_of_week'],

              hue=uber_april_2014[uber_april_2014['Base']=='Weiter']['Hour'],palette='seismic')

plt.title('Weiter Base',size=30)

plt.legend(loc=(1.05,0))

plt.show()

# here the traffic is less on saturday sunday and monday

# the trend is same no exception
# Insight: Schmecken Base traffic monitoring on the bases of Day_of_week and Hours 

plt.figure(figsize=(16,8))

sns.countplot(uber_april_2014[uber_april_2014['Base']=='Schmecken']['Day_of_week'],

              hue=uber_april_2014[uber_april_2014['Base']=='Schmecken']['Hour'],palette='seismic')

plt.title('Schmecken Base',size=30)

plt.legend(loc=(1.05,0))

plt.show()

# here the traffic is less on sunday and monday

# the trend is same no exception
# Insight: Danach_NY Base traffic monitoring on the bases of Day_of_week and Hours 

plt.figure(figsize=(16,8))

sns.countplot(uber_april_2014[uber_april_2014['Base']=='Danach_NY']['Day_of_week'],

              hue=uber_april_2014[uber_april_2014['Base']=='Danach_NY']['Hour'],palette='seismic')

plt.title('Danach_NY Base',size=30)

plt.legend(loc=(1.05,0))

plt.show()

# here the traffic is less on sunday and monday

# the trend is same no exception
# comparing between base

plt.figure(figsize=(10,5))

sns.countplot(uber_april_2014['Base'],palette='viridis')

# Schmecken,Weiter and Hinter base have more pickups as compared to Unter and Danach_NY base