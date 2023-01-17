# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import plotly.plotly as py
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import mpld3
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df.head()
train_df = pd.read_csv('../input/metro-bike-share-trip-data.csv', low_memory = False)

#print(train_df.shape[0])
#train_df = train_df.dropna()
#print(train_df.shape[0])
train_df.head()
#train_df.plot(x='Plan Duration', y = 'Duration', kind='kde')

train_df.Duration = train_df.Duration.apply(lambda x: x/60)

plt.figure(figsize=(10,6))
plt.xlabel("Duration in minutes")
plt.ylabel("Count")
train_df.Duration.hist(bins=20, range=(0,120))
train_df.Duration.describe()
    

format = "%Y-%m-%dT%H:%M:%S"
start_time = pd.to_datetime(train_df["Start Time"], format=format)
time = pd.DatetimeIndex(start_time)
start_hour = pd.DataFrame(time.hour + time.minute/60)

fig, ax = plt.subplots(figsize=(16, 8))
for ph in ["Monthly Pass", "Flex Pass", "Walk-up"]:
    d = start_hour[train_df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.25)
    plt.xlabel("Hour")
plt.title("Plan Usage throughout the Day")
plt.legend()


for ph in ["Monthly Pass", "Flex Pass", "Walk-up"]:
    fig, ax = plt.subplots(figsize=(16, 8))
    d = start_hour[train_df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.25)
    plt.xlabel("Hour")
    plt.title(ph + " Usage throughout the Day")


start_station_occurences = train_df.groupby('Starting Station ID').size()
#print(start_station_occurences)
stop_station_occurences = train_df.groupby('Ending Station ID').size()
start_station_occurences.idxmax()
stop_station_occurences.idxmax()
from math import sin, cos, sqrt, atan2, radians

R = 6371 #radius of earth in KM

lat1 = train_df["Starting Station Latitude"].apply(radians)
lon1 = train_df["Starting Station Longitude"].apply(radians)
lat2 = train_df["Ending Station Latitude"].apply(radians)
lon2 = train_df["Ending Station Longitude"].apply(radians)

dlat = lat2 - lat1
dlon = lon1 - lon2

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

train_df["Distance"] = R * c
dist = R * c
train_df["Distance"].describe()
print(train_df['Distance'].median())
pass_type_occurences = train_df.groupby('Passholder Type').size()
print(pass_type_occurences)
average_distance_df = train_df.groupby('Starting Station ID').mean()

q = average_distance_df["Distance"].quantile(0.99)
average_distance_df = average_distance_df[average_distance_df['Distance'] < q]
print("The Starting Station with the greatest average trip distance is Station ID " + str(average_distance_df['Distance'].idxmax()) + " with an average of " + str(average_distance_df['Distance'].max()))
average_distance_df = average_distance_df.reset_index()
average_distance_df.head()

fig, ax = plt.subplots(figsize=(24, 8))
plt.bar(average_distance_df['Starting Station ID'], average_distance_df['Distance'])


plt.xlabel("Starting Station")
plt.ylabel("Average Distance (km)")

q = train_df["Distance"].quantile(0.99)
d = train_df[train_df['Distance'] < q]
d = d.groupby('Passholder Type').mean()

print(d)
d = d.reset_index()
labels = list(d['Passholder Type'])
values = list(d['Distance'])
fig, ax = plt.subplots(figsize=(24, 8))
plt.bar(labels, values)
plt.ylabel("Average Distance (km)")



temp = pd.DataFrame(data=[train_df['Duration'], train_df['Starting Station Latitude'],train_df['Starting Station Longitude'],train_df['Ending Station Latitude'],
                            train_df['Ending Station Longitude'],train_df['Plan Duration']],index=['Duration','Starting Station Latitude','Starting Station Longitude',
                                    'Ending Station Latitude','Ending Station Longitude','Plan Duration'])
train = temp.T
train = train.reset_index(drop=True)
train = pd.concat([dist,train, pd.get_dummies(data=train_df['Passholder Type']).reset_index(), pd.get_dummies(data=train_df['Trip Route Category'],drop_first=True).reset_index()], axis=1)

train = train.drop('index',axis=1)
train = train.dropna()
X = train.drop(columns=['Flex Pass','Monthly Pass','Walk-up','Plan Duration'])
Y = train['Walk-up']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size =0.7,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
Y_pred= clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print(accuracy_score(Y_pred, Y_test))
print()
print(classification_report(Y_pred, Y_test))