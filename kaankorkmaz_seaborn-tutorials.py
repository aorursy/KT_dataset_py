# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading datas

percentage_people_below_poverty_level = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

median_house_hold_in_come = pd.read_csv("../input/MedianHouseholdIncome2015.csv",encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

    

data = pd.DataFrame({"area_list": area_list,"area_poverty_ratio": area_poverty_ratio})

new_index = (data["area_poverty_ratio"].sort_values(ascending = False).index.values)

sorted_data = data.reindex(new_index)



#visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["area_list"], y= sorted_data["area_poverty_ratio"])

plt.xticks(rotation = 45)

plt.xlabel("States")

plt.ylabel("Poverty Rate")

plt.title("Poverty Rate Given States")

plt.show()
kill.head()
kill.name.value_counts()
seperate = kill.name[kill.name != "TK TK"].str.split()

a,b = zip(*seperate)

name_list = a+b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)

#visualization

plt.figure(figsize=(15,10))

sns.barplot(x=x,y=y, palette= sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most Common 15 Name Or Surname Of Killed People")

plt.show()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

# sorting

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share_race_city.info()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share_race_city['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))



# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='yellow',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='black',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='green',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='blue',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='higher right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

plt.show()
#highschool graduation vs poverty rate of each state

sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_data2["area_highschool_ratio"] = sorted_data2["area_highschool_ratio"]/max(sorted_data2["area_highschool_ratio"])

data = pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1)

data.sort_values("area_poverty_ratio",inplace=True)

#visualization

plt.subplots(figsize=(20,10))

sns.pointplot(x="area_list",y="area_highschool_ratio",data=data,color="blue",alpha=0.9)

sns.pointplot(x="area_list",y="area_poverty_ratio",data=data,color="black",alpha=0.8)

plt.text(40,0.35,"poverty ratio",color="black",fontsize=18,style="oblique")

plt.text(40,0.30,"area highschool ratio",color="blue",fontsize=18,style="oblique")

plt.xlabel("States",fontsize=15,color="red")

plt.ylabel("Values",fontsize=15,color="red")

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='red')

plt.grid()