# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading deep sea coral data
coral = pd.read_csv('../input/deep_sea_corals.csv')
# Lets see our data's columns
coral.columns
# First 5 row as a sample from the data
coral.head(2)
# Remove the first row because its 'NaN'
#coral.drop([0], inplace=True)
coral = coral.dropna()
coral.head(2)
# To examine how many entries are there in the data:
coral.info()
# In order to see how many times each species is observed in the data:
#print(coral.ScientificName.value_counts())
print('TOTAL NUMBER OF OBSERVED CORAL SPECIES = ', len(coral.ScientificName.value_counts()))
# Lets turn it into a new data frame:
dataSpeciesCount = pd.DataFrame(coral.ScientificName.value_counts())
# Beacuse there are 2888 rows in this data, I want take first 50 species for my new data set.
dataSpeciesCount2 = dataSpeciesCount.head(50)
dataSpeciesCount2.head()
# Our new dataset's index is species names. But i want to have a seperate column for species names:
dataSpeciesCount2['Species Name'] = dataSpeciesCount2.index 
# And change the index name as number:
dataSpeciesCount2.index = range(1,51,1)
# Change the column name of "ScientificName" to "Total Observations"
dataSpeciesCount2.rename(columns={'ScientificName':'Total Observations'}, inplace=True)
#Change the order of the columns:
newdata = dataSpeciesCount2.reindex(columns=['Species Name', 'Total Observations'])
newdata.head()
# BARPLOT EXAMPLE
plt.figure(figsize=(15,10))
ax = sns.barplot(x=newdata['Total Observations'], y=newdata['Species Name'])
ax.set(xlabel='Total Number Of Observations', ylabel='Observed Species', title='Most Observed 50 Coral Species')
coral.head(2)
# Making a new dataset made of vernacular name categories and depths (concatination example)
coral01 = pd.concat([coral.VernacularNameCategory, coral.DepthInMeters], axis=1)
# Sorting values from deepest to shallowest
coral01.sort_values('DepthInMeters', ascending=False, inplace=True)
# Filtering wrong depth values
filter01 = coral01.DepthInMeters > 0
coral01 = coral01[filter01]
# Indexing again
coral01.index = range(1, 1+len(coral01))
coral01.head()
# Making another dataset to see number of observations for each vernatular name category
coral02 = pd.DataFrame(coral01.VernacularNameCategory.value_counts())
coral02.rename(columns={'VernacularNameCategory':'NuOfObservations'}, inplace=True)
coral02['VernacularNameCategory']= coral02.index
coral02.index = range(1, 1+len(coral02))
coral02
# Showing speciy observations according to the vernacular name category (count plot example) 
plt.figure(figsize=(15,6))
sns.countplot(coral.VernacularNameCategory)
plt.grid()
plt.title('Observations of Species According to Their Vernacular Name Category', fontsize=15, color='red')
plt.xlabel('Vernacular Name Categories of the Species')
plt.ylabel('Number of Observations')
plt.xticks(rotation='45')
plt.show()
# Showing relationship of coral obervation numbers and observed depths
f,ax = plt.subplots(figsize=(20,5))
sns.scatterplot(x='VernacularNameCategory', y='DepthInMeters', data=coral01, color='blue')
sns.pointplot(x=coral02.VernacularNameCategory, y=coral02.NuOfObservations/10, color='orange') 
plt.text(11,4500, 'Observed Depths', color='blue', fontsize=15, style='italic')
plt.text(11,4200, 'Number of Observations (Divided by 10)', color='orange', fontsize=15, style='italic')
plt.xlabel('Vernacular Name Categories of Corals', color='blue', fontsize=13)
plt.ylabel('Depths and Observation Numbers', color='blue', fontsize=13)
plt.title('Coral Species Relationship between Number of Observations and Observed Depths', color='red', fontsize=14)
plt.xticks(rotation=45)
plt.grid()
# How many different sampling equipments do we have?
coral.SamplingEquipment.value_counts()
# Showing sampling equipments percentage on a pie chart:
labels = coral.SamplingEquipment.value_counts().index
colors = ['orange', 'yellow', 'pink', 'red', 'purple', 'black']
explode = [0,0.05,0.1,0.15,0.2,0.25]
sizes = coral.SamplingEquipment.value_counts().values

plt.figure(figsize=(11,11))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=225, textprops={'fontsize':15})
plt.title('Sampling Equipments Used for Coral Observations', color='blue', fontsize=15)
# Creating a new dataset:
a = coral.DataProvider
b = coral.VernacularNameCategory
c = coral.DepthInMeters
d = coral.SamplingEquipment
coral04 = pd.concat([a, b, c, d], axis=1)
coral04.head(2)
# Data providers contribution and depths they collected the data from. (boxplot example)
plt.figure(figsize=(20,10))
sns.boxplot(x='DataProvider', y='DepthInMeters', data=coral04, palette='PRGn')
plt.xticks(rotation=90)
plt.title('Data Providers Contribution and Collected Data Depth', fontsize=14, color='red')
plt.xlabel('Name of the Data Providers')
plt.ylabel('Depths of the Data (meters)')
plt.show()
# LETS ANALYSE SEA PEN SPECIES ACCORDING TO DATA PROVIDERS, SAMPLING TOOLS AND OBSERVATION DEPTHS
# Filtering data to analyse only sea pens:
filter03 = coral04.VernacularNameCategory == 'sea pen'
coral05 = coral04[filter03]
coral05.head()
# Sea Pen Observations According to Data Providers, Sampling Equipments and Depth (swarmplot example)
plt.figure(figsize=(20,10))
sns.swarmplot(x='SamplingEquipment', y='DepthInMeters', hue='DataProvider', data=coral05)
plt.grid()
plt.xticks(rotation='45')
plt.title('Sea Pen Observations According to Data Providers, Sampling Equipments and Depth', fontsize=15, color='red')
plt.xlabel('Sampling Equipments')
plt.ylabel('Depths of the Data (meters)')
plt.show()