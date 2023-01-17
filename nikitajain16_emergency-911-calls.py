# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('/kaggle/input/montcoalert/911.csv')

df.head()
df.shape
df.info()
df.isnull().sum() #finding out null values for each column
df.duplicated().sum() #finding out duplcated rows if any
df.drop_duplicates(keep=False,inplace=True) #dropping the duplicate rows

df.duplicated().sum()
df.describe()
#creating a copy of original dataset

df_clean=df.copy()

df_clean.head()
#renaming the columns

df_clean.rename(columns = {"lat": "latitude", 

                           "lng":"longitude", 

                           "desc": "description",

                            "twp":"township",

                            "addr":"address",

                            "timeStamp":"time"},inplace=True)

df_clean.columns
#changing the type of zip code column and timestamp column

df_clean['zip'] = df_clean['zip'].astype(str)

df_clean['time']=pd.to_datetime(df_clean['time'])

df_clean.head()

df_clean.info()
#removing the decimal point from zip column

def change_zip(x):

    x=x[0:5]

    return x

df_clean['zip']= df_clean['zip'].apply(change_zip)

df_clean.head()
#extracting reason from title

def make_reason(x):

    x=x.split(':')[0]

    return x

df_clean['Reason']= df_clean['title'].apply(make_reason)

df_clean.head()
#extracting month,day,year,hour from timestamp column

df_clean['Hour']= df_clean['time'].apply(lambda t: t.hour)

df_clean['Month']= df_clean['time'].apply(lambda t: t.month)

df_clean['Day of Week']= df_clean['time'].apply(lambda t: t.strftime('%A'))

df_clean['Year']= df_clean['time'].apply(lambda t: t.year)

df_clean['Date']= df_clean['time'].apply(lambda x: x.date())

df_clean.head()
df_clean.groupby("Reason")['e'].count()
df_clean['township'].unique()
df_clean['zip'].unique()
df_clean.groupby(['Date','Reason'])['e'].count()
plt.figure(figsize=(8,4), dpi=80)

lat=df_clean[(df_clean['latitude']>39) & (df_clean['latitude']<41)]['latitude']

plt.hist(lat);

plt.xlabel("LATITUDE")

plt.ylabel("COUNT");

plt.title("Distribution of latitude");

plt.show()
plt.figure(figsize=(8,4), dpi=80)

long=df_clean[(df_clean['longitude']>-76) & (df_clean['longitude']<-74)]['longitude']

plt.hist(lat);

plt.xlabel("LONGITUDE")

plt.ylabel("COUNT");

plt.title("Distribution of longitude");
plt.figure(figsize=(8,4), dpi=80)

sns.countplot(x='Reason',data=df_clean);

plt.title("Reason for 911 Calls");
order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

plt.figure(figsize=(8,4), dpi=80)

sns.countplot(x='Day of Week', hue='Reason',data=df_clean,order=order);

plt.title("Number of 911 Calls Per Day");

plt.legend(loc=3);
plt.figure(figsize=(8,4), dpi=80)

sns.countplot(x='Month', hue='Reason',data=df_clean);

plt.title("Number of 911 Calls per month")

plt.legend(loc=3);
plt.figure(figsize=(8,4), dpi=80)

sns.countplot(x='Year', hue='Reason',data=df_clean);

plt.title("Number of 911 Calls per year")

plt.legend(loc=2);
#month

plt.figure(figsize=(8,4), dpi=80)

month=df_clean.groupby('Month').count()

plt.plot(month['e'])

plt.xlabel("Number of 911 Calls per month");

plt.ylabel("Count");

plt.title("911 Calls per month");



#day

plt.figure(figsize=(8,4), dpi=80)

month=df_clean.groupby('Day of Week').count()

plt.plot(month['e'])

plt.xlabel("Day");

plt.ylabel("Count");

plt.title("911 Calls per day");



#hour

plt.figure(figsize=(8,4), dpi=80)

month=df_clean.groupby('Hour').count()

plt.plot(month['e'])

plt.xlabel("Hour of the Day");

plt.ylabel("Count");

plt.title("911 Calls per hour");
plt.figure(figsize=(8,4), dpi=80)

dayHour = df_clean.groupby(['Day of Week','Hour']).count().unstack()['Reason']

sns.heatmap(dayHour)
plt.figure(figsize=(20,50), dpi=200)

sns.countplot(y='township',hue='Reason',data=df_clean);