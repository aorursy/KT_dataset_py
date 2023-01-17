# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import folium

from folium.plugins import FastMarkerCluster

from folium.plugins import MarkerCluster

from sklearn.cluster import DBSCAN

from folium.plugins import HeatMap

# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv("../input/ny-permit-issuance/DOB_Permit_Issuance.csv", nrows=100000) #Trying to maintain the computational sanity of my device with nrows. 

                                                                                            #please increase limits if you are using more powerful device

ratdata = pd.read_csv("../input/nyc-rat-sightings/Rat_Sightings.csv")
dataframe.head(5)
dataframe.dtypes
ax = dataframe.groupby('BOROUGH')['BOROUGH'].count().sort_values().plot(

    kind='bar', figsize=(10,6), title="Permit count by Borough")

for p in ax.patches:

    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()))
ax = dataframe.groupby('Permit Status')['Permit Status'].count().sort_values().plot(

    kind='barh',  figsize=(10,8), title="Number of Permits Issued")
dataframe.groupby('Permit Type')['Permit Type'].count().sort_values().plot(kind='bar', figsize=(15,8),

                                                                          title="Comparing types of Permits")
res = dataframe[dataframe['Residential']=='YES']['Residential'].count()

nres = dataframe['Residential'].isna().sum()

data = {'residential':res,'Non-Residential':nres}

# print(data)

typedf = pd.DataFrame(data = data,index=['Counts'])

# typedf.head()

typedf.plot(kind='barh', title="Residential Vs Non Residential Permits")
dataframe['Issuance Date'] = pd.to_datetime(dataframe['Issuance Date'])
dataframe['issued year'] = dataframe['Issuance Date'].dt.year
dataframe['issued month'] = dataframe['Issuance Date'].dt.month
dataframe.head()
monthDF = dataframe[dataframe['Permit Status']=='ISSUED'].groupby('issued month')['issued month'].count()

monthDF.plot(kind='bar', title="Number of permits by Month", figsize=(8,6))
fig, axes = plt.subplots(nrows=2, ncols=2)

monthDF2019 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)].groupby('issued month')['issued month'].count()

monthDF2018 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)].groupby('issued month')['issued month'].count()



# .plot(

# kind='bar', ax=axes[0,0])

monthDF2019.plot(kind='bar', title="Number of permits by Month 2019", ax=axes[0,0], figsize=(10,8))

monthDF2018.plot(kind='bar', title="Number of permits by Month 2018", ax=axes[0,1], figsize=(10,8))

fig.delaxes(axes[1][1])

fig.delaxes(axes[1][0])
borough2018 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)]['BOROUGH']

borough2019 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)]['BOROUGH']





fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes

width = 0.4



dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)]['BOROUGH'].value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)

dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)]['BOROUGH'].value_counts().plot(kind='bar', color='orange', ax=ax, width=width, position=0)

ax.set_ylabel('Counts')

plt.legend(['2018', '2019'], loc='upper right')

ax.set_title('2018 Vs 2019 Borough Building Permit Counts')
print("There are ",dataframe['LATITUDE'].isna().sum(), "Missing location data")

dataframe = dataframe.dropna(subset=['LATITUDE','LONGITUDE'])

print("Number of missing data:", dataframe['LATITUDE'].isna().sum())
NYmap = folium.Map(location=[40.7128,-74.0060],zoom_start=10)

NYmap
mc = MarkerCluster()

for row in dataframe[0:10000].itertuples():

    mc.add_child(folium.Marker(location =[row.LATITUDE,row.LONGITUDE],popup = row.BOROUGH))
NYmap.add_child(mc)

NYmap
ratdata.head()
ratdata = ratdata[ratdata['Borough'] != 'Unspecified']
fig, axes = plt.subplots(nrows=2, ncols=2)

ratdata.groupby('Borough')['Borough'].count().plot(kind='bar', title="Rat Sightings by Borough", ax = axes[0,1], figsize=(12,10))

dataframe.groupby('BOROUGH')['BOROUGH'].count().plot(kind='bar', title='Building Permit By Borough', ax=axes[0,0], figsize=(12,10))

fig.delaxes(axes[1][1])

fig.delaxes(axes[1][0])
ratdata = ratdata.dropna(subset=['Latitude', 'Longitude'])
heat_data = [[row['Latitude'],row['Longitude']] for index, row in ratdata.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(NYmap)

NYmap