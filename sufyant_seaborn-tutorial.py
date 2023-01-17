import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level = percentage_people_below_poverty_level[percentage_people_below_poverty_level['poverty_rate'] != '-']

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)



poverty_rate = (percentage_people_below_poverty_level.groupby('Geographic Area').sum()

                /percentage_people_below_poverty_level.groupby('Geographic Area').count()

                [['poverty_rate']]).sort_values('poverty_rate',ascending=False)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=poverty_rate.index, y=poverty_rate['poverty_rate'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
kill['name'].value_counts()
kill = kill[kill.name != 'TK TK']

kill = kill[kill.name != 'TK Tk']
kill[kill['name'].duplicated()]
seperate = kill['name'].str.split

last_name = []

first_name = []

for i in seperate():

    last_name.append(i[-1])

    first_name.append(i[0])

kill['last_name'] = pd.DataFrame(last_name)

kill['first_name'] = pd.DataFrame(first_name)
kill.head()
x = kill.groupby('first_name').count()[['id']].sort_values('id',ascending=False)[0:15]



plt.figure(figsize=(15,10))

ax= sns.barplot(x=x.index, y=x.id,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 First Name of killed people')
y=kill.groupby('last_name').count()[['id']].sort_values('id',ascending=False)[0:15]



plt.figure(figsize=(15,10))

ax= sns.barplot(x=y.index, y=y.id,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name of killed people')
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool['percent_completed_hs'].value_counts()
percent_over_25_completed_highSchool = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['percent_completed_hs'] != '-']

percent_over_25_completed_highSchool['percent_completed_hs'] = percent_over_25_completed_highSchool['percent_completed_hs'].astype(float)

x = (percent_over_25_completed_highSchool.groupby('Geographic Area').sum()/percent_over_25_completed_highSchool.groupby('Geographic Area').count()).sort_values('percent_completed_hs',ascending=True)[['percent_completed_hs']]

x
plt.figure(figsize=(15,10))

sns.barplot(x=x.index,y=x['percent_completed_hs'])

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.xticks(rotation= 45)
share_race_city.head()
share_race_city.info()
share_race_city.isna().sum()
# drop the value

share_race_city = share_race_city[share_race_city['share_white'] != '(X)']



# change to dtype of columns

share_race_city['share_white'] = share_race_city['share_white'].astype(float)

share_race_city['share_black'] = share_race_city['share_black'].astype(float)

share_race_city['share_native_american'] = share_race_city['share_native_american'].astype(float)

share_race_city['share_asian'] = share_race_city['share_asian'].astype(float)

share_race_city['share_hispanic'] = share_race_city['share_hispanic'].astype(float)





x = (share_race_city.groupby('Geographic area').sum()/share_race_city.groupby('Geographic area').count())
plt.subplots(figsize = (9,15))

sns.barplot(x=x['share_white'],y=x.index,color='green',alpha = 0.5,label='White' )

sns.barplot(x=x['share_black'],y=x.index,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=x['share_native_american'],y=x.index,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=x['share_asian'],y=x.index,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=x['share_hispanic'],y=x.index,color='red',alpha = 0.6,label='Hispanic')



plt.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

plt.xlabel('Percentage of Races')

plt.ylabel('States')

plt.title("Percentage of State's Population According to Races")
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool['percent_completed_hs'].value_counts().index.unique()
percent_over_25_completed_highSchool = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['percent_completed_hs'] != '-']

percent_over_25_completed_highSchool['percent_completed_hs'] = percent_over_25_completed_highSchool['percent_completed_hs'].astype(float)

x = (percent_over_25_completed_highSchool.groupby('Geographic Area').sum()/percent_over_25_completed_highSchool.groupby('Geographic Area').count()).sort_values('percent_completed_hs',ascending=True)[['percent_completed_hs']]

x.head()
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level['poverty_rate'].value_counts().index.unique()
percentage_people_below_poverty_level = percentage_people_below_poverty_level[percentage_people_below_poverty_level['poverty_rate'] != '-']

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)



y = (percentage_people_below_poverty_level.groupby('Geographic Area').sum()

                /percentage_people_below_poverty_level.groupby('Geographic Area').count()

                [['poverty_rate']]).sort_values('poverty_rate',ascending=False)

y.head()
# high school graduation rate vs Poverty rate of each state

y['poverty_rate'] = y['poverty_rate']/max( y['poverty_rate'])

x['percent_completed_hs'] = x['percent_completed_hs']/max( x['percent_completed_hs'])

data = pd.concat([y,x['percent_completed_hs']],axis=1)

data.sort_values('poverty_rate',inplace=True)



# visualize

plt.subplots(figsize =(20,10))

sns.pointplot(x=data.index,y='poverty_rate',data=data,color='cyan',alpha=0.8)

sns.pointplot(x=data.index,y='percent_completed_hs',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17)

plt.text(40,0.55,'poverty ratio',color='cyan',fontsize = 18)

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
data.head()
data.tail()
data.corr()
g = sns.jointplot(data['poverty_rate'], data['percent_completed_hs'], kind="kde", height=7)
g = sns.jointplot(data['poverty_rate'], data['percent_completed_hs'],size=7,ratio=3,color='r')
kill.head()
kill['race'].value_counts()
labels = kill['race'].value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
percent_over_25_completed_highSchool = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['percent_completed_hs'] != '-']

percent_over_25_completed_highSchool['percent_completed_hs'] = percent_over_25_completed_highSchool['percent_completed_hs'].astype(float)

x = (percent_over_25_completed_highSchool.groupby('Geographic Area').sum()/percent_over_25_completed_highSchool.groupby('Geographic Area').count()).sort_values('percent_completed_hs',ascending=True)[['percent_completed_hs']]



percentage_people_below_poverty_level = percentage_people_below_poverty_level[percentage_people_below_poverty_level['poverty_rate'] != '-']

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)



y = (percentage_people_below_poverty_level.groupby('Geographic Area').sum()

                /percentage_people_below_poverty_level.groupby('Geographic Area').count()

                [['poverty_rate']]).sort_values('poverty_rate',ascending=False)



# high school graduation rate vs Poverty rate of each state

y['poverty_rate'] = y['poverty_rate']/max( y['poverty_rate'])

x['percent_completed_hs'] = x['percent_completed_hs']/max( x['percent_completed_hs'])

data = pd.concat([y,x['percent_completed_hs']],axis=1)

data.sort_values('poverty_rate',inplace=True)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

sns.lmplot(x="poverty_rate", y="percent_completed_hs", data=data)

plt.show()
data.head()
sns.kdeplot(data.poverty_rate, data.percent_completed_hs, shade=True, cut=3);
data.head()
plt.figure(figsize = (7,7))

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points");
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax);
kill.head()
kill.manner_of_death.unique()
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="Set1");
kill.head()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill);
data.head()
sns.pairplot(data);
kill.head()
kill.gender.value_counts()
sns.countplot(kill.gender)

plt.title("gender",color = 'blue',fontsize=15)
armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15);
# age of killed people

above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15);
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15);
# Most dangerous cities

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15);
# most dangerous states

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15);
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15);
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15);
# Threat types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15);
# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15);
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15);
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15);