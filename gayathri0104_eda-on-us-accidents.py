import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
input_data_dir = "../input/us-accidents"

input_data_file = "US_Accidents_May19.csv"

input_data_path = os.path.join(input_data_dir, input_data_file)

acci = pd.read_csv(input_data_path)
pd.set_option('display.max_columns',None)
acci = acci.set_index('ID')
acci.isnull().sum()
acci.shape
acci = acci[~acci.duplicated()]
acci.shape
missing = pd.DataFrame((acci.isnull().sum()*100/len(acci)).sort_values(ascending=False),columns=['% percent'])
missing[missing['% percent']>15]
acci.drop(['Number','End_Lat','End_Lng','Wind_Chill(F)','Precipitation(in)'],1,inplace=True)
acci.shape
acci.head()
acci.select_dtypes(include=['float','int']).head()
acci.TMC.value_counts()
acci.TMC.mean()
acci.TMC = acci.TMC.fillna(207)
acci['Wind_Speed(mph)'].mean()
acci['Wind_Speed(mph)'] = acci['Wind_Speed(mph)'].fillna(8.8)
acci = acci.dropna()
acci.shape
acci.isnull().sum().sum()
acci.select_dtypes(exclude=['float','int']).head()
acci.Source.value_counts()
acci.Severity.value_counts()
from datetime import datetime as dt
acci.Start_Time = acci.Start_Time.astype('datetime64')

acci.End_Time = acci.End_Time.astype('datetime64')
acci['Hour'] = acci.Start_Time.dt.hour
acci['Diff_in_min'] =  acci.End_Time - acci.Start_Time 
acci.Diff_in_min = acci.Diff_in_min.astype('timedelta64[m]')
acci.Timezone.nunique()
acci.drop(['Source','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight',

           'Description','Side','Zipcode','County','Airport_Code','Weather_Timestamp'],1,inplace=True)
acci.head()
plt.figure(figsize=(15,6))

acci.TMC.value_counts().plot(kind='bar')
acci.State.nunique()
plt.figure(figsize=(15,6))

acci.State.value_counts().plot(kind='bar',color='r')
plt.figure(figsize=(25,6))

sns.countplot(acci.State,hue=acci.Severity)
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

acci.City[acci.State=='CA'].value_counts().head().plot(kind='bar',color='y')



plt.subplot(1,2,2)

acci.City.value_counts().head().plot(kind='bar',color='m')
plt.figure(figsize=(15,6))

round((acci.Diff_in_min/60),0).value_counts().head().plot(kind='bar',color='g')
plt.figure(figsize=(15,6))

sns.countplot(acci.Hour)
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

acci.Wind_Direction.value_counts().head().plot(kind='bar',color='m')



plt.subplot(1,2,2)

acci.Weather_Condition.value_counts().head().plot(kind='bar',color='c')
plt.figure(figsize=(10,5))



plt.subplot(2,2,1)

acci.Junction.value_counts().plot(kind='pie')



plt.subplot(2,2,2)

acci['Sunrise_Sunset'].value_counts().plot(kind='pie')



plt.subplot(2,2,3)

acci.Traffic_Signal.value_counts().plot(kind='pie')

acci.Bump.value_counts()