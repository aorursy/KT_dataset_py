# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv',encoding = "windows-1252")

percent_over_25_completed_highschool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv',encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv',encoding="windows-1252")

police_kill = pd.read_csv('../input/PoliceKillingsUS.csv',encoding="windows-1252")
percentage_people_below_poverty_level.head(20)

#First 5 row about Data.
percentage_people_below_poverty_level.info()

#General info about Data
percentage_people_below_poverty_level.poverty_rate.value_counts()

#Value count at poverty_rate columns
percentage_people_below_poverty_level.poverty_rate.replace(('-'),0.0, inplace=True)

#Value='-' replace, Value='0'
percentage_people_below_poverty_level.poverty_rate.value_counts()

#Checked the Values
percentage_people_below_poverty_level.poverty_rate  = percentage_people_below_poverty_level.poverty_rate.astype(float)

#I need int value, but object(string) in the poverty_rate. Data type transformed. 
percentage_people_below_poverty_level.info()

#checked data type.
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())



#Unique Geopraphic Area...

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list, 'area_poverty_ratio': area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



#visualization

plt.figure(figsize=(20,15))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
police_kill.head()
police_kill.name.value_counts()
#seperate = police_kill.name[police_kill.name != 'TK TK'].str.split()

#a,b = zip(*seperate)

#name_list = a+b

#name_count = name_list

#most_common_names = name_count.most_common(15)

#x,y = zip(*most_common_names)

#x,y = list(x),list(y)



#plt.figure(figsize=(20,20))

#ax= sns.barplot(x=x, y=y, palette  = sns.cubehelix_palette(len(x)))

#plt.xlabel('Name or Surname of killed people')

#plt.ylabel('Frequency')

#plt.title('Most common 15 people Name or Surname of killed people')

percent_over_25_completed_highschool.info()
percent_over_25_completed_highschool.head()
percent_over_25_completed_highschool.percent_completed_hs.value_counts()

percent_over_25_completed_highschool.percent_completed_hs.replace(['-'],0.0, inplace=True)

percent_over_25_completed_highschool.percent_completed_hs = percent_over_25_completed_highschool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highschool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highschool[percent_over_25_completed_highschool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

data = pd.DataFrame({'area_list': area_list, 'area_highschool_ratio': area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values

sorted_data2 = data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")



share_race_city.head()
share_race_city.info()
share_race_city.replace(['_'],0.0,inplace=True)

share_race_city.replace(['x'],0.0,inplace=True)
