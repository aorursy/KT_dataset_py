# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(['ls', '../input']).decode("utf8"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percent_over_25_completed_highSchool.head()
share_race_city.head()
kill.head()
#Poverty rate of each state

percentage_people_below_poverty_level['Geographic Area'].unique()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)     
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio=[]

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list' : area_list, 'area_poverty_ratio' : area_poverty_ratio})

new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)
plt.figure(figsize=(15,10))

ax=sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xlabel('states')

plt.ylabel('poverty rate')

plt.title('Poverty rate given States')
data.head()
kill.name.value_counts()
#Most Common 15 Name or Surname of killed person

separate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*separate)

name_list = a+b

name_count=Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)
plt.figure(figsize=(15,10))

ax = sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel("name and surname")

plt.ylabel("frequency")

plt.title("most common 15 name or surname of killed person")
# Percentage of state's population according to races that are black,white,native american, asian and hispanic
share_race_city.head()
share_race_city.info()
share_race_city.replace(['-'],0.0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
share_race_city.loc[:,["share_white", "share_black", "share_native_american","share_asian","share_hispanic"]].astype(float)              
area_list = list(share_race_city['Geographic area'].unique())
area_list
share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]
for i in area_list:

    x=share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white.astype(float))/len(x))

    share_black.append(sum(x.share_black.astype(float))/len(x))

    share_native_american.append(sum(x.share_native_american.astype(float))/len(x))

    share_asian.append(sum(x.share_asian.astype(float))/len(x))

    share_hispanic.append(sum(x.share_hispanic.astype(float))/len(x))

#visualization

f,ax = plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list,color='g',label='white',alpha=0.7)

sns.barplot(x=share_black,y=area_list,color='b',label='black',alpha=0.7)

sns.barplot(x=share_native_american,y=area_list,color='c',label='native',alpha=0.7)

sns.barplot(x=share_asian,y=area_list,color='y',label='asian',alpha=0.7)

sns.barplot(x=share_hispanic,y=area_list,color='r',label='hispanic',alpha=0.7)



ax.legend(loc='best',frameon=True)#frameon=visibility of frame

ax.set(xlabel='percentage of races', ylabel='states',

       title="Percentage of State's Population According to Races ")
