# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import folium

from folium.plugins import MarkerCluster

# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv', parse_dates=['Reported Date'])
dataframe.head()
dataframe.dtypes
dataframe.groupby('Reported Date')['Reported Date'].count().plot(kind='line', title="Incident Reports Graph", figsize=(14,6))
dataframe.groupby('Reported Year')['Reported Year'].count().plot(kind='bar', title="Number of incidents based on year", figsize=(14,6))
monthstocheck=['Jan','Feb','Mar','Apr','May','Jun']

# data = dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts()

# dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().sort_index(ascending=False).plot(kind='barh', title="Comparison First six months of 2019 with previous years", figsize=(10,6))

plt.figure(figsize=(10,6))

sns.barplot(dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().index,dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().values, alpha=0.8)

plt.ylabel("Number of Incidents")

plt.xlabel("Reported Year")

plt.title("Comparing first six months of 2019 with previous years")
dataframe.isna().sum()
dataframe.groupby('Reported Year')['Number Dead'].sum().astype(int).plot(kind='bar', title="Number of Deaths based on Year", figsize=(10,6))
dataframe.groupby('Reported Year')['Total Dead and Missing'].sum().astype(int).plot(kind='barh', title="Number of Dead/Missing per Year", figsize=(10,6))
df = dataframe.groupby('Reported Month').agg({'Number of Females':'sum','Number of Males':'sum'}).astype(int)

# x = dataframe['Reported Month'].value_counts().index.values

# y = dataframe['Reported Month'].value_counts().values

plt.figure(figsize=(12,6))

sns.lineplot(data=df)
dataframe.groupby('Reported Year').agg({'Number of Females':'sum','Number of Males':'sum'}).astype(int).plot(kind='bar',stacked=True, title="Number of Missing/Dead Female Vs Male", figsize=(10,6))
dataframe.head()
wordcloud = WordCloud(

    width = 1000,

    height = 600,

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(dataframe['Cause of Death']))



fig = plt.figure(

    figsize = (20, 12))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

# plt.tight_layout(pad=0)

plt.show()
dataframe['Cause of Death'].value_counts()[0:15].plot(kind='bar', title="Fifteen common causes of Death", figsize=(15,8))
dataframe['Region of Incident'].value_counts().plot(kind='barh', title="Number of Incident report based on Region", figsize=(15,10))
dataframe.head()
dataframe.groupby('Region of Incident')['Total Dead and Missing'].sum().sort_values().plot(kind='bar', title="Dead/Missing Based of the Region", figsize=(12,8))
dataframe['Migration Route'].value_counts().sort_values().plot(kind='bar', title="Most common Routes for migrants", figsize=(14,6))
dataframe[['latitude','longitude']] = dataframe['Location Coordinates'].str.split(",",expand=True)
dataframe['latitude'] = dataframe['latitude'].astype(float).round(2)

dataframe['longitude'] = dataframe['longitude'].astype(float).round(2)
dataframe.head()
dataframe = dataframe.dropna(subset=['latitude','longitude'])
dataframe.isna().sum()
worldMap = folium.Map(zoom_start=16)
worldMap
mc = MarkerCluster()

for row in dataframe.itertuples():

    mc.add_child(folium.Marker(location=[row.latitude,row.longitude]))
worldMap.add_child(mc)