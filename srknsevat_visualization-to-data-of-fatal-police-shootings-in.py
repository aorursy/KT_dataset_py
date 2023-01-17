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

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

percentage_people_below_poverty_level.poverty_rate.value_counts()
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique()) 

print(area_list)
area_poverty_ratio = []  # We have defined an empty list

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

# Sorting Data - Sıralanmış Liste

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

sorted_data.head()
plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
kill.info()
kill.head()
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split() 

separate
a,b = zip(*separate)                  

name_list = a+b                         

name_count = Counter(name_list)      

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)
plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')

plt.show()
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

print(area_list)

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

print(area_highschool)
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

sorted_data2.head()
plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share_race_city.head()
share_race_city.info()
# share_race_city.share_white.value_counts()

# share_race_city.share_black.value_counts()

# share_race_city.share_native_american.value_counts()

# share_race_city.share_asian.value_counts()

# share_race_city.share_hispanic.value_counts()
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

    

print(area_list)

    

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

plt.show()
Normalization
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

print(sorted_data.head(),sorted_data2.head())
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

data.head()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3)

plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
data.corr()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
sns.pairplot(data)

plt.show()
kill.head()
kill.race.value_counts()
kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values
plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

plt.show()
kill.manner_of_death.unique()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

# Plot the orbital period with horizontal boxes

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()
# swarm plot

# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()
kill.gender.value_counts()
kill.head()
sns.countplot(kill.gender)

plt.title("gender",color = 'blue',fontsize=15)

plt.show()
armed = kill.armed.value_counts()

armed
plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)

plt.show()
above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

df.head()
sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)

plt.show()
sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)

plt.show()
city = kill.city.value_counts()

city.head()
plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)

plt.show()
state = kill.state.value_counts()

state.head()
plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)

plt.show()
sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)

plt.show()
sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)

plt.show()
sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)

plt.show()
sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)

plt.show()
sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)

plt.show()