import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv',encoding = 'windows-1252')

percentage_people_below_povert_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding = 'windows-1252')

percent_over_25_completed_HighSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding = 'windows-1252')

share_race_by_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding = 'windows-1252')

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding = 'windows-1252')
percentage_people_below_povert_level.head()
percentage_people_below_povert_level.info()
# Poverty Rate of Each State

percentage_people_below_povert_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_povert_level.poverty_rate = percentage_people_below_povert_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_povert_level['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_povert_level[percentage_people_below_povert_level['Geographic Area'] == i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list' : area_list,'area_poverty_ratio' : area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)
# Visualizatoin

plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data['area_list'],y = sorted_data['area_poverty_ratio'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
# Most Common 15 Names and Surnames Killed People

seperate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*seperate)

name_list = a + b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)
# Visualization

plt.figure(figsize=(15,10))

sns.barplot(x = x,y = y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Names')

plt.ylabel('Frequency')

plt.title('Most Common 15 Name or Surname of Killed People')

plt.show()
percent_over_25_completed_HighSchool.head()
percent_over_25_completed_HighSchool.info()
percent_over_25_completed_HighSchool.percent_completed_hs.value_counts()
percent_over_25_completed_HighSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_HighSchool.percent_completed_hs = percent_over_25_completed_HighSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_HighSchool['Geographic Area'].unique())

area_HighSchool = []

for i in area_list:

    x = percent_over_25_completed_HighSchool[percent_over_25_completed_HighSchool['Geographic Area'] == i]

    area_HighSchool_rate = sum(x.percent_completed_hs)/len(x)

    area_HighSchool.append(area_HighSchool_rate)

data = pd.DataFrame({'area_list' : area_list,'area_highschool_ratio' : area_HighSchool})

new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values

sorted_data2 = data.reindex(new_index)
plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data2['area_list'],y = sorted_data2['area_highschool_ratio'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share_race_by_city.head()
share_race_by_city.info()
share_race_by_city.share_white.value_counts()

share_race_by_city.share_black.value_counts()

share_race_by_city.share_native_american.value_counts()

share_race_by_city.share_asian.value_counts()

share_race_by_city.share_hispanic.value_counts()
# Percentage of State's Population According due to races that are Black, White,Native,American,Asian and Hispanic

share_race_by_city.replace(['-'],0.0,inplace = True)

share_race_by_city.replace(['(X)'],0.0,inplace = True)

share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share_race_by_city['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]

for i in area_list:

    x = share_race_by_city[share_race_by_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))
# Visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x = share_white,y = area_list,alpha = 0.5,color = 'green',label = 'White')

sns.barplot(x = share_black,y = area_list,alpha = 0.5,color = 'blue',label = 'African American')

sns.barplot(x = share_native_american,y = area_list,alpha = 0.5,color = 'cyan',label = 'Native American')

sns.barplot(x = share_asian,y = area_list,alpha = 0.5,color = 'yellow',label = 'Asian')

sns.barplot(x = share_hispanic,y = area_list,alpha = 0.5,color = 'red',label = 'Hispanic')

ax.legend(loc = 'lower right',frameon = True)

plt.xlabel('Percentage of Races')

plt.ylabel('States')

plt.title("Percentage of State's Population According due to races that are Black, White,Native,American,Asian and Hispanic")

plt.show()
# High School Graduation Rate vs Poverty Rate of Each State

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis = 1)

data.sort_values('area_poverty_ratio',inplace = True)
# Visualization

plt.subplots(figsize = (20,10))

sns.pointplot(data = data, x = 'area_list',y = 'area_poverty_ratio',color = 'lime',alpha = 0.8)

sns.pointplot(data = data, x = 'area_list', y = 'area_highschool_ratio',color = 'red',alpha = 0.8)

plt.text(38,0.6,'High School Graduate Ratio',color = 'red',fontsize = 17,style = 'italic')

plt.text(38,0.55,'Poverty Ratio',color = 'lime',fontsize = 17,style = 'italic')

plt.xlabel('Values',color = 'blue',fontsize = 15)

plt.ylabel('States',color = 'blue',fontsize = 15)

plt.title('High School Graduation Rate vs Poverty Rate of Each State',color = 'blue',fontsize = 17)

plt.grid()

plt.show()
sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind = 'kde',size = 7)

plt.show()
sns.jointplot('area_poverty_ratio','area_highschool_ratio',data = data,size = 5,ratio = 3,color = 'red')

plt.show()
# Race Rates according in Kill Data

kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values
# Visualization

plt.figure(figsize = (7,7))

plt.pie(sizes,explode = explode,labels = labels,colors = colors,autopct = '%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

plt.show()
sns.lmplot('area_poverty_ratio','area_highschool_ratio',data = data)

plt.show()
# High School Graduation Rate vs Poverty Rate of Each State

sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade = True,cut = 5)

plt.show()
# High School Graduation Rate vs Poverty Rate of Each State

sns.violinplot(data = data,inner = 'points')

plt.show()
# Correlation Map

# High School Graduation Rate vs Poverty Rate of Each State

# annot for the looking numbers

plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(),annot = True,linewidth = 0.5,fmt = '.1f')

plt.show()
kill.head()
# Manner of Death : shot and tasered

# gender and age

sns.boxplot(x = 'gender',y = 'age',hue = 'manner_of_death',data = kill,palette = 'PRGn')

plt.show()
# High School Graduation Rate vs Poverty Rate of Each State

sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death',data = kill)

plt.show()
# Pair Plot

sns.pairplot(data)

plt.show()
# Kill Weapon

armed = kill.armed.value_counts()

plt.figure(figsize = (10,7))

sns.barplot(x = armed[:7].index,y = armed[:7].values)

plt.xlabel('Weapon Types')

plt.ylabel('Number of Weapon')

plt.title('Kill Weapon',color = 'blue',fontsize = 15)

plt.show()
# Race of Killed People

sns.countplot(data = kill,x = 'race')

plt.title('Race of Killed People')

plt.show()