# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
percentOver25CompletedHighschool=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding='windows-1252')

medianHousesholdIncome2015=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv',encoding='windows-1252')

ShareRaceByCity=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding='windows-1252')

PoliceKillingsUS=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding='windows-1252')

PercentagePeopleBelowPovertyLevel=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding='windows-1252')


# Poverty rate of each state



PercentagePeopleBelowPovertyLevel.head()
PercentagePeopleBelowPovertyLevel.info()
PercentagePeopleBelowPovertyLevel['Geographic Area'].unique()
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'],0.0,inplace=True)

PercentagePeopleBelowPovertyLevel.poverty_rate=PercentagePeopleBelowPovertyLevel.poverty_rate.astype(float)

area_list=list(PercentagePeopleBelowPovertyLevel['Geographic Area'].unique())

area_poverty_ratio = []



for i in area_list:

    x=PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel['Geographic Area']==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
# Most common 15 Name or Surname of killed people





PoliceKillingsUS.head()
PoliceKillingsUS.name.value_counts()

#49 'TK TK' is not name
separate=PoliceKillingsUS.name[PoliceKillingsUS.name!='TK TK'].str.split() 

a,b=zip(*separate)



name_list=a+b











name_count=Counter(name_list)



most_common_names=name_count.most_common(15)

x,y=zip(*most_common_names)

x,y=list(x),list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most Common 15 name or surname of killed people')

percentOver25CompletedHighschool.head()
percentOver25CompletedHighschool.info()
percentOver25CompletedHighschool.percent_completed_hs.replace(['-'],0.0,inplace=True)

percentOver25CompletedHighschool.percent_completed_hs=percentOver25CompletedHighschool.percent_completed_hs.astype(float)

area_list =list(percentOver25CompletedHighschool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x=percentOver25CompletedHighschool[percentOver25CompletedHighschool['Geographic Area']==i]

    area_highschool_rate=sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

data=pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})

new_index =(data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)

#visiulation



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation=90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")


