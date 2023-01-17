# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from collections import Counter





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())

area_poverty_ratio=[]

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})

new_index=(data["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)

#visualisiation

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("poverty_rate")

plt.title("Poverty_rate_given_States")
kill.head()
kill.info()
kill.name.value_counts()
seperate=kill.name[kill.name!="TK TK"].str.split()

a,b=zip(*seperate)

name_list=a+b

name_count=Counter(name_list)

most_common_names=name_count.most_common(15)

x,y=zip(*most_common_names)

x,y=list(x),list(y)

plt.figure(figsize=(15,10))

sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname killed people")

plt.ylabel("Frequency")

plt.title("Most common killed peoople name")
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list2=list(percent_over_25_completed_highSchool["Geographic Area"].unique())

area_ratio2=[]

for i in area_list:

    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    area_rate2=sum(x.percent_completed_hs)/len(x)

    area_ratio2.append(area_rate2)

data2=pd.DataFrame({"area_list":area_list2,"area_ratio":area_ratio2})

new_index2=(data2["area_ratio"].sort_values(ascending=True)).index.values

sorted_data2=data2.reindex(new_index2)





plt.figure(figsize=(10,15))

sns.barplot(x=sorted_data2["area_list"],y=sorted_data2["area_ratio"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("percent_completed")

plt.title("Perecent_completed_hs of States")

    