# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv',encoding = "windows-1252")
house_hold_income = pd.read_csv('../input/MedianHouseholdIncome2015.csv',encoding = "windows-1252")
completed_highschool_percent = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv',encoding = "windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv',encoding = "windows-1252")
share_race = pd.read_csv('../input/ShareRaceByCity.csv',encoding = "windows-1252")
#Biggest median income of according state
house_hold_income['Median Income'].replace(['(X)'],0.0,inplace=True)
house_hold_income['Median Income'].replace(['-'],0.0,inplace = True)
house_hold_income['Median Income'].replace(['2,500-'],0.0,inplace = True)
house_hold_income['Median Income'].replace(['250,000+'],0.0,inplace = True)
house_hold_income['Median Income'] = house_hold_income['Median Income'].astype(float)
house_hold_income[house_hold_income['Median Income'] == max(house_hold_income['Median Income'])]

people_below_poverty_level.info()
people_below_poverty_level.head()
people_below_poverty_level.poverty_rate.unique()
#Poverty rate of each state
people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)
people_below_poverty_level.poverty_rate = people_below_poverty_level.poverty_rate.astype(float)
area_list = list(people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio = []

for i in area_list:
    x = people_below_poverty_level[people_below_poverty_level['Geographic Area'] == i]
    #were taking areas ratio
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

#visuluzation

plt.figure(figsize = (15,10))
sns.barplot(x=sorted_data.area_list,y=sorted_data.area_poverty_ratio)
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Area Poverty Ratio')
plt.title('Poverty Rate Given States')
plt.show()

kill.name
#Most common 15 Name or Surname of killed People
#in data we know tk tk
seperate = kill.name[kill.name != 'TK TK'].str.split()
a,b = zip(*seperate)
name_list = a+b
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
x,y = zip(*most_common_names)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
ax = sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))
plt.xlabel('Name of Killed People')
plt.ylabel('Frequency')
plt.title('Most common 15 Name of Killed People')
plt.show()
#High School Graduation rate of the population that is older than 25 in states
completed_highschool_percent.percent_completed_hs.replace(['-'],0.0,inplace=True)
completed_highschool_percent.percent_completed_hs = completed_highschool_percent.percent_completed_hs.astype(float)

high_school_ratio=[]
for i in area_list:
    x = completed_highschool_percent[completed_highschool_percent['Geographic Area']==i]
    high_school_rate = sum(x.percent_completed_hs)/len(x)
    high_school_ratio.append(high_school_rate)
    
data = pd.DataFrame({'area_list':area_list,'high_school_ratio':high_school_ratio})
new_index = (data['high_school_ratio'].sort_values(ascending=True)).index.values
sorted_data2 =data.reindex(new_index)

#visualize

plt.figure(figsize =(15,10))
ax = sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['high_school_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('High School Graduate Ratio')
plt.title('Percentage of Given States Population Above 25 That Has Graduated High School')
plt.show()

share_race.head()
#Percentage of state's population according to races that are black,white,native american,asian and hispanic
share_race.replace(['-'],0.0,inplace=True)
share_race.replace(['(X)'],0.0,inplace=True)
share_race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic=[]
for i in area_list:
    x = share_race[share_race['Geographic area'] == i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))
    
#visulization
f, ax = plt.subplots(figsize = (9,15))
sns.barplot(x=share_white,y=area_list,color="green",alpha=0.5,label="White")
sns.barplot(x=share_black,y=area_list,color="blue",alpha=0.5,label="Afro american")
sns.barplot(x=share_native_american,y=area_list,color="cyan",alpha=0.5,label="Native american")
sns.barplot(x=share_asian,y=area_list,color="yellow",alpha=0.5,label="Asian")
sns.barplot(x=share_hispanic,y=area_list,color="red",alpha=0.5,label="Hispanic")

ax.legend(loc='lower right',frameon=True)
ax.set(xlabel='Percentage of Races',ylabel='States',title='Percentage of States Population according to Races')
plt.show()

sorted_data2.head()
sorted_data.head()
#High school graduation rate vs Poverty rate of each state

#normalize the data
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
sorted_data2['high_school_ratio'] = sorted_data2['high_school_ratio']/max(sorted_data2['high_school_ratio'])

data = pd.concat([sorted_data,sorted_data2['high_school_ratio']],axis=1) #concat with columns
data.sort_values('area_poverty_ratio',inplace=True)

#visualize

f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='high_school_ratio',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'High School Graduate Ratio',color='red',fontsize =20,style='normal')
plt.text(40,0.55,'Poverty Ratio',color = 'lime',fontsize = 20,style='normal')
plt.xlabel('States',fontsize = 20,color='blue')
plt.ylabel('Values',fontsize=20,color='blue')
plt.title('High School Graduate VS Poverty Rate',fontsize = 20,color='blue')
plt.grid()
plt.show()
#Visualization of High School Graduation rate vs Poverty rate of each state with Different style of seaborn code
#joint kernel density
#pearsonr : if = 1 positive correlation and if =-1 there is negative correlation
#if its zero there is no correlation between variables
#Show the joint distribution using kernel density estimation

g = sns.jointplot(data.area_poverty_ratio,data.high_school_ratio,kind='kde',size=7)
plt.savefig('graph.png')
plt.show()
#you can change paramters of joint plot

g = sns.jointplot(data.area_poverty_ratio,data.high_school_ratio,size=5,ratio=3,color='r')
kill.head()
#Race Rates According to kill data
kill.race.dropna(inplace=True)
labels=kill.race.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = kill.race.value_counts().values

#visualize

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.2f%%')
plt.title("Killed People According To Races",color = "blue",fontsize = 20)
plt.show()
#Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
#lmplot
#Show the results of a linear regression within each dataset

sns.lmplot(x= 'area_poverty_ratio',y ='high_school_ratio',data=data)
plt.show()
#Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code
#cubehelix plot
sns.kdeplot(data.area_poverty_ratio,data.high_school_ratio,shade=True,cut=3)
plt.show()
#Show each distribution with bot violins and points
#Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2,rot=-.5,dark=.3)
sns.violinplot(data=data,palette=pal,inner = 'points')
plt.show()
#data correlation
data.corr()
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,linewidths =.5,fmt='.3f',ax=ax)
plt.show()
kill.head()
#manner of death : shot , shot and tasered
#Plot the orbital period horizontal boxes
sns.boxplot(x='gender',y='age',hue='manner_of_death',data=kill,palette='PRGn')
plt.show()
#swarm plot
#manner of death 
sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=kill)
plt.title('MANNER OF DEATH',color='black',fontsize=20)
plt.show()
data.head()
#pair plot of data
sns.pairplot(data)
plt.show()
#Kill properties
#Manner of Death
#Count plot
sns.countplot(kill.gender)
sns.countplot(kill.manner_of_death)
plt.title('Manner of Death',color='black',fontsize=20)
plt.show()
#kill weapon
armed = kill.armed.value_counts()
plt.figure(figsize=(10,10))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.ylabel('Number Of Weapon')
plt.xlabel('Weapon Types')
plt.title('Homiside Weapon',color="black",fontsize=20)
plt.show()
#age of killed people above 25 or below 25
#countplot
above25 = ['above25' if i>=25 else 'below25' for i in kill.age]
df = pd.DataFrame({'age':above25})
sns.countplot(x=df.age)
plt.ylabel('Number Of Killed People')
plt.title('Age of Killed People',color="black",fontsize=20)
#Most dangerous cities
city = kill.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:12].index,y=city[:12].values)
plt.xticks(rotation=45)
plt.title('Most Dangerous cities',fontsize=20)
#Race of killed people
sns.countplot(data=kill,x='race')
plt.title('Race Of Killed People',color="black",fontsize=20)
plt.show()
#Having mental ilness or not for killed people
sns.countplot(kill.signs_of_mental_illness)
plt.xlabel('Mental Illness')
plt.ylabel('Number of Mental ilness')
plt.title('Having mental ilness Or Not',fontsize=20)
plt.show()
