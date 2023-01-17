# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap
path = "../input/crimes-in-boston/crime.csv"

df = pd.read_csv(path,encoding = 'unicode_escape')
df.head()
df.info()
feat_desc = pd.DataFrame({'Description': df.columns, 

                          'Values': [df[i].unique() for i in df.columns],

                          'Number of unique values': [len(df[i].unique()) for i in df.columns]})

feat_desc
df.rename(columns = {"INCIDENT_NUMBER": "Incident_Number", 

                     "OFFENSE_CODE":"Offense_Code","OFFENSE_CODE_GROUP":"Offense_Code_Group","OFFENSE_DESCRIPTION":"Offense_Description",

                     "DISTRICT": "District","REPORTING_AREA": "Reporting_Area","SHOOTING": "Shooting",

                     "OCCURRED_ON_DATE": "Occurred_On_Date","YEAR": "Year","MONTH": "Month",

                     "DAY_OF_WEEK": "Day_Of_Week","HOUR": "Hour","UCR_PART": "Ucr_Part",

                     "STREET": "Street"

                     }, 

                                 inplace = True) 
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
# Fill in nans in SHOOTING column

df.Shooting.fillna('N', inplace=True)
df=df.dropna()
def getSeason(month):

    if (month == 12 or month == 1 or month == 2):

       return "WINTER"

    elif(month == 3 or month == 4 or month == 5):

       return "SPRING"

    elif(month ==6 or month==7 or month == 8):

       return "SUMMER"

    else:

       return "FALL"
df['Season'] = df.Month.apply(getSeason)
fig, ax = plt.subplots(figsize=(17,8))

with sns.color_palette("RdBu",4):

  montyearAggregated = pd.DataFrame(df.groupby(["Month","Year"])["Incident_Number"].count()).reset_index()

  a=sns.barplot(data=montyearAggregated,x="Month", y="Incident_Number",hue = 'Year')

  a.set_title("Crime",fontsize=15)

  plt.legend(loc='upper right')

  plt.show()
df_year_new=df.groupby(["Year","Month"])["Incident_Number"].count().reset_index()

df_year_filter=df_year_new[~df_year_new['Month'].isin(['1','2','3','4','5','11','12'])]

fig = plt.figure(figsize=(8,6))

with sns.color_palette("RdBu",4):

  b=sns.countplot(x="Year",data=df_year_filter)

  b.set_title("Crime",fontsize=14)

  plt.show();
fig = plt.figure(figsize=(8,8))

with sns.color_palette("RdBu",4):

  ctplt2=sns.catplot(x="Year", y="Incident_Number",kind="box", data=df_year_new,size=5, aspect=2)

  plt.ylabel('Count')

  plt.show();
fig,axes= plt.subplots(2,2)

fig.set_size_inches(16,12)

with sns.color_palette("RdBu",4):

  a=sns.countplot(x="Day_Of_Week",order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],data=df,ax=axes[0, 0])

  a.set(xlabel='Dayofweek', ylabel='Total Crime')

  a.set_title("Total Crime Amount By Weekday",fontsize=15)



  b=sns.countplot(x="Month",data=df,ax=axes[0, 1])

  b.set(xlabel='Month', ylabel='Total Crime')

  b.set_title("Total Crime Amount By Month",fontsize=15)



  c=sns.countplot(x="Season",data=df,ax=axes[1, 0])

  c.set(xlabel='Season', ylabel='Total Crime')

  c.set_title("Total Crime Amount By Season",fontsize=15)



  d=sns.countplot(x="Year",data=df,ax=axes[1, 1])

  d.set(xlabel='Year', ylabel='Total Crime')

  d.set(xlabel='Year', ylabel='Total Crime')

  d.set_title("Total Crime Amount By Year",fontsize=15);
# Displaying Top 20 street that have a high crime

fig = plt.figure(figsize=(12,5))

crime_street = df.groupby('Street')['Incident_Number'].count().nlargest(20)

crime_street.plot(kind='bar')

plt.xlabel("Street")

plt.ylabel("Offense Amount")

plt.show()
df_location=df.groupby(['Month','Year','District'])['Incident_Number'].count().reset_index()
ctplt=sns.catplot(x="District", y="Incident_Number",kind="box", data=df_location,size=10, aspect=2)

plt.ylabel('Offense Amount');
with sns.color_palette("RdBu",4):

  fig = plt.figure(figsize=(12,5))

  order = df['Offense_Code_Group'].value_counts().head(10).index

  chart=sns.countplot(data = df, x='Offense_Code_Group', order = order);

  plt.xlabel("Offense Group")

  plt.ylabel("Offense Amount")

  chart.set_xticklabels(chart.get_xticklabels(), rotation=90);

fig = plt.figure(figsize=(20,10))

order2 = df['Offense_Code_Group'].value_counts().head(5).index

sns.countplot(data = df, x='Offense_Code_Group',hue='District', order = order2);

plt.ylabel("Offense Amount");
df_map=df.groupby(['Lat','Long'])['Incident_Number'].count().reset_index()

df_map.head()
import plotly.express as px

x=df_map['Lat']

y=df_map['Long']

fig=px.density_mapbox(df_map, lat='Lat', lon='Long', z='Incident_Number', radius=12,color_continuous_scale="Portland",

                      center=dict(lat=42.319945, lon=-71.079989), zoom=10,

                      mapbox_style='stamen-terrain')

fig.show()