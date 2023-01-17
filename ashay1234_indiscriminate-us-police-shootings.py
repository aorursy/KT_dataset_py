# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## loading the data into dataframe
PercentOver25CompletedHighSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding = 'ISO-8859-1')

PoliceKillingsUS = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding = 'ISO-8859-1')

PercentagePeopleBelowPovertyLevel = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding = 'ISO-8859-1')
## doing data cleaning of percent_completed_hs column
PercentOver25CompletedHighSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
PercentOver25CompletedHighSchool.percent_completed_hs = PercentOver25CompletedHighSchool.percent_completed_hs.astype(float)

## fetching unique Geographic Area value. making a list of it 
area_list = list(PercentOver25CompletedHighSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:
    x = PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)

## creating a dataframe of people > 25 yrs in age and having completed high school
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

# sorting according to area_highschool_ratio
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
## reindexing so that our index starts once agian from 0 to n-1
sorted_data2 = data.reindex(new_index)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
## below graph describes lot of insigths, but still does not give a clear indication 
## that whether the low rate of education in certain states is the root cause of the indiscriminate shootings
# Race rates according in kill data 
PoliceKillingsUS.race.dropna(inplace = True)
labels = PoliceKillingsUS.race.value_counts().index
colors = ['purple','green','blue','yellow','orange','grey']
explode = [0,0,0,0,0,0]
sizes = PoliceKillingsUS.race.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
## The comparison of the poverty rate among all States

## data cleaning
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'],0.0,inplace = True)
PercentagePeopleBelowPovertyLevel.poverty_rate = PercentagePeopleBelowPovertyLevel.poverty_rate.astype(float)

area_list = list(PercentagePeopleBelowPovertyLevel['Geographic Area'].unique())
area_poverty_ratio = []

for i  in area_list:
    x = PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

# Respresenting the data in visual form
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
# Most dangerous cities
city = PoliceKillingsUS.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:12].index,y=city[:12].values)
plt.xticks(rotation=45)
plt.title('Most dangerous cities',color = 'blue',fontsize=15)

## mapping the most dangerous cities in order to find a correlation.

## Is their direct correlation between the dangers in the cities 
## leading to increased crime rates and finally the police shootings ??
#finding a correlation of the fact whether there is a correlation between the mental health and the killings of the victims
sns.countplot(PoliceKillingsUS.signs_of_mental_illness)
plt.xlabel('Mental illness')
plt.ylabel('Number of Mental illness')
plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
#finding a correlaion beween the threat level of the intended victims and their killings
sns.countplot(PoliceKillingsUS.threat_level)
plt.xlabel('Threat Types')
plt.title('Threat types',color = 'blue', fontsize = 15)