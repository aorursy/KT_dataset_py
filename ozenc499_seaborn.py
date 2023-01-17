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
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level["Geographic Area"].unique()
sum(percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']=='AL'].poverty_rate)/len(percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']=='AL'])
area_list = percentage_people_below_poverty_level["Geographic Area"].unique()

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

    
(data['area_poverty_ratio'].sort_values(ascending=False)).index.values
data['area_poverty_ratio'].sort_values(ascending=False).index.values
plt.figure(figsize=(15,10))

ax2 = sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

#plt.xlabel('States')

#plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

ax2.set(xlabel='States', ylabel='Poverty Rate')

plt.show()
kill.head()
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split()

separate
a,b= zip(*separate)

name_list = a+b

name_count = Counter(name_list) 

#name_count
most_common_names = name_count.most_common(15) 

most_common_names
x,y = zip(*most_common_names)

x = list(x)

y = list(y)
plt.figure(figsize=(15,10))

ax3 = sns.barplot(x=x, y=y)

plt.title('Most common 15 Name or Surname of killed people')

ax3.set(xlabel='Name or Surname of killed people', ylabel='Frequency')

plt.show()

percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace('-',0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
percent_over_25_completed_highSchool.info()
area_list = list(percent_over_25_completed_highSchool["Geographic Area"].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

data = pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})    

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xlabel = 'States'

plt.ylabel = 'Highschool Ratio'

plt.show()

    
share_race_city.info()
share_race_city.tail()
#share_race_city['share_white'].value_counts()

#share_race_city['share_black'].value_counts()

#share_race_city['share_native_american'].value_counts()

#share_race_city['share_asian'].value_counts()

#share_race_city['share_hispanic'].value_counts()
share_race_city.replace('(X)',0.0,inplace=True)


share_race_city.share_white = share_race_city.share_white.astype(float)

share_race_city.share_black=share_race_city.share_black.astype(float)

share_race_city.share_native_american=share_race_city.share_native_american.astype(float)

share_race_city.share_asian=share_race_city.share_asian.astype(float)

share_race_city.share_hispanic=share_race_city.share_hispanic.astype(float)





share_race_city.info()
share_race_city.loc[1]
# area list will be one of our arguments for our dictionary. We will use this list as y axis in our graph.

area_list = list(share_race_city['Geographic area'].unique())
len(area_list)
len(share_race_city)
share_race_city.info()
# len(area_list) olmayabilir onun yerine AL de kac whiteín kac sehire orani olabilir.

# alpha argument is important. It makes the multiple graphics are possiable to be seen.
white = []

black = []

native = []

asian = []

hispanic = []

for i in area_list:

    white.append(sum(share_race_city.share_white[share_race_city['Geographic area'] == i])/len(share_race_city.share_white[share_race_city['Geographic area'] == 'AL']))

    black.append(sum(share_race_city.share_black[share_race_city['Geographic area'] == i])/len(share_race_city.share_white[share_race_city['Geographic area'] == 'AL']))

    native.append(sum(share_race_city.share_native_american[share_race_city['Geographic area'] == i])/len(share_race_city.share_white[share_race_city['Geographic area'] == 'AL']))

    asian.append(sum(share_race_city.share_asian[share_race_city['Geographic area'] == i])/len(share_race_city.share_white[share_race_city['Geographic area'] == 'AL']))

    hispanic.append(sum(share_race_city.share_hispanic[share_race_city['Geographic area'] == i])/len(share_race_city.share_white[share_race_city['Geographic area'] == 'AL']))

f,ax = plt.subplots(figsize = (9,15))

#alpha saydamlik demek

sns.barplot(x=white,y=area_list,label='white',alpha = 0.5, color = 'cyan')

sns.barplot(x=black,y=area_list,label='black',alpha = 0.5, color = 'green')

sns.barplot(x=native,y=area_list,label='native',alpha = 0.5, color = 'yellow')

sns.barplot(x=asian,y=area_list,label='asian',alpha = 0.5, color = 'red')

sns.barplot(x=hispanic,y=area_list,label='hispanic',alpha = 0.5, color = 'orange')

ax.legend(loc='lower right',frameon = True)

ax.set(xlabel='Percentage of the races', ylabel='Area List',title='Ratio of the races according to areas')

# we normalize the sorted_date set as below.

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
# if we check the data as below,  we can see that is recalculated based on the max value. 

sorted_data.head()
#normalizing the sorted_date2. So that we can compare them.

sorted_data2['area_highschool_ratio']= sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])
sorted_data2.head()
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)
data.sort_values('area_poverty_ratio',inplace=True)
data.head()
f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x = 'area_list', y = 'area_poverty_ratio',data=data,color='blue',alpha=0.8)

sns.pointplot(x = 'area_list', y = 'area_highschool_ratio',data=data,color='orange',alpha=0.8)

plt.text(40,0.6,'High School Graduation',color='orange',fontsize = 17,style='italic')

plt.text(40,0.55,'Poverty',color='blue',fontsize = 17,style='italic')

ax1.set(xlabel='States', ylabel='Normalized Values')

plt.title('Graduation vs Poverty')

plt.grid()

data.head()
g = sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind='kde',size=7)

plt.savefig('figure.png')

plt.show()

sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data,size = 5, ratio=3)
kill.race.unique()
# we dropped the values N/A

kill.race.dropna(inplace=True)
kill.race.unique()
kill.race.value_counts()
labels= kill.race.value_counts().index
labels
sizes = kill.race.value_counts().values
sizes
colors = ['red','blue','pink','orange','lime','cyan']

explode = [0,0,0,0,0,0]

plt.figure(figsize = (7,7))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Kill Rates by Races',color='blue',fontsize=15)

sns.lmplot(x='area_poverty_ratio',y = 'area_highschool_ratio',data=data )

plt.show()
pal = sns.cubehelix_palette(2,rot = -.5,dark=.3)

sns.violinplot(data=data,palette=pal,inner="points")

plt.show()
data.corr()
data.head()
#plt.figure(figsize=(5,5))

f,ax = plt.subplots(figsize = (4,4))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

plt.show()
kill.head()
sns.boxplot(x="gender",y="age", hue="manner_of_death",data = kill,palette="PRGn")
sns.swarmplot(x="gender",y="age", hue="manner_of_death",data = kill)

plt.show()
kill.head()
kill.manner_of_death.value_counts()
#sns.countplot(kill.gender)

sns.countplot(kill.manner_of_death)

plt.title('manner of death',color='blue',size=15)

plt.show()
armed = kill.armed.value_counts()

plt.figure(figsize=(10,7))

ax = sns.barplot(x=armed[:7].index, y=armed[:7].values )

ax.set(xlabel='Armed', ylabel='Number of weapons')

plt.show()
above25 = ['above25' if i >=25 else 'below25' for i in kill.age]
df = pd.DataFrame({'age':above25})

ax = sns.countplot(x=df.age)

ax.set(xlabel='Age', ylabel='Number of killed people')