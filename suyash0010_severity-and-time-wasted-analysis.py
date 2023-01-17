# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
plt.rcParams.update({'font.size': 22,'legend.fontsize': 10})

plt.style.use('seaborn-dark')

%matplotlib inline
df=pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv',parse_dates=['Start_Time','End_Time'])

df.head()
len(df)
df.info()
df.isnull().sum()
df= df.drop(['Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','Precipitation(in)','Wind_Direction','Pressure(in)','Humidity(%)','Wind_Chill(F)','Temperature(F)','Weather_Timestamp','Airport_Code','Timezone','Zipcode','Number','End_Lat','End_Lng','TMC'],axis=1)
df['Visibility(mi)'].fillna(method='bfill',inplace=True)

df['Weather_Condition'].fillna(method='bfill',inplace=True)

df['Sunrise_Sunset'].fillna(method='bfill',inplace=True)

df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].mean(),inplace=True)

df['City'].fillna(df['City'].mode().iloc[0],inplace=True)

df['Description'].fillna(df['Description'].mode().iloc[0],inplace=True)
df.isnull().sum()
duration_hour = (df['End_Time']-df['Start_Time'])

my_hour = []

for each in pd.to_datetime(duration_hour):

    my_hour.append(each.hour)

duration_df =pd.DataFrame({'hour':my_hour})

hourwise= duration_df['hour'].value_counts()

hourwise.sort_index(inplace=True)
hourwise
accidents_more_than_five = hourwise[5:].sum()

hourwise.drop(hourwise[5:].index,inplace=True)

hourwise['>5'] = accidents_more_than_five

hourwise
plt.subplots(figsize=(15,10))

plt.grid(True)

sns.barplot(x=hourwise.values,y=hourwise.index)

plt.xticks(rotation = 60)
duration_df['City']= df['City']

duration_df.head()
df_sum = duration_df.groupby('City')['hour'].sum()

df_sum.sort_values(ascending=False,inplace=True)

df_top = df_sum.head(30)

plt.subplots(figsize=(20,10))

sns.barplot(x=df_top.index,y = df_top.values)

plt.xticks(rotation=60)
boolean_type=[]

for column in df.columns:

    if df[column].dtype=='bool':

        boolean_type.append(column)

boolean_type
def creating_groups(col):

    df_new = df.groupby([col,'Severity'])['ID'].count()

    df_new = df_new.unstack('Severity')

    df_new = df_new.fillna(0)

    return df_new

for columns in boolean_type:

    print(creating_groups(columns))
df_list=[]

def  bool_severity(col):

    df_bool = df.groupby([col,'Severity'])['ID'].count()

    df_bool = df_bool.unstack(col)

    df_list.append(df_bool)

for col in boolean_type:

    bool_severity(col)

for i in range(len(boolean_type)):

    plt.tight_layout()

    df_list[i].plot(kind='pie',subplots=True,autopct='%.1f%%',pctdistance=1.4,figsize=(10,10),title = boolean_type[i])

df['Side']=df['Side'].replace(' ','R')

df_s_s = df.groupby(['Side','Severity'])['ID'].count()

df_s_s = df_s_s.unstack('Severity')

df_s_s.plot(kind='pie',subplots=True,figsize=(20,10),autopct='%.1f%%')

plt.xticks(rotaion=90)
df_dis = df.groupby('Severity')['Distance(mi)'].mean()

df_dis

plt.subplots(figsize=(15,10))

sns.barplot(x=df_dis.index,y=df_dis.values)