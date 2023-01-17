# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/metro-bike-share-trip-data.csv")
data.info()
data.head()
data.columns = data.columns.str.replace(' ','_')
data.head()
data['Start_Time']=pd.to_datetime(data['Start_Time'])
data['End_Time']=pd.to_datetime(data['End_Time'])
data.info()
data = data.dropna()
data.info()
data['Starting_Station_ID'] = data.Starting_Station_ID.astype(int)
data['Ending_Station_ID'] = data.Ending_Station_ID.astype(int)
data['Bike_ID'] = data.Bike_ID.astype(int)
data.head()
print(data.Starting_Station_ID.nunique())
print(data.Starting_Station_Latitude.nunique())
print(data.Starting_Station_Longitude.nunique())
print(data.Ending_Station_ID.nunique())
print(data.Ending_Station_Latitude.nunique())
print(data.Ending_Station_Longitude.nunique())
data.Bike_ID.nunique()
data.Trip_Route_Category.value_counts().plot(kind='bar')
plt.show()
data.Passholder_Type.value_counts().plot(kind='pie',autopct='%.2f')
plt.show()
data.Plan_Duration.value_counts().plot(kind  ='pie', autopct='%.2f')
plt.show()
data.Start_Time.dt.date.nunique()
print(data.Start_Time.min())
print(data.Start_Time.max())
data.Bike_ID.value_counts().head()
data.Starting_Station_ID.value_counts().head()
data.Ending_Station_ID.value_counts().head()
data.Duration.value_counts().head()
print(data.Duration.min())
print(data.Duration.max())
time_filter = data.Duration<2*60*60
data[time_filter].Duration.value_counts().plot(kind='bar', figsize=(10,10))
plt.show()
data.Start_Time.dt.hour.value_counts().plot(kind='bar')
plt.show()

Monthly = data.Passholder_Type == "Monthly Pass"
Walk = data.Passholder_Type == "Walk-up"
Flex = data.Passholder_Type == "Flex Pass"
fig = plt.figure
plt.subplot(1,2,1)
plt.bar(("monthly","walk-up","free"),[data[Monthly].Duration.mean(),data[Walk].Duration.mean(),data[Flex].Duration.mean()])
plt.subplot(1,2,2)
plt.pie([data[Monthly].Duration.sum(), data[Walk].Duration.sum(), data[Flex].Duration.sum()], labels = ("monthly","walk-up","free"), autopct='%.2f')
plt.show()

data["duration_level"] = ["long" if i>4*60*60 else "very short" if i<=10*60 else "short" if i<=60*60  else "mid" for i in data.Duration]
data.duration_level.value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()