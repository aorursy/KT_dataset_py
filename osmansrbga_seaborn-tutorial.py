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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read datas

median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv',encoding = "windows-1252")

percentage_people_below_poverty_level = pd.read_csv ('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding = "windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding = "windows-1252")

share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding = "windows-1252")

kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding = "windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()
percentage_people_below_poverty_level['Geographic Area'].value_counts()
#BAR PLOT

#Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list' : area_list, 'area_poverty_ratio' :area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('STATES')

plt.ylabel('POVERTY RATE')

plt.title('POVERTY RATE GİVEN STATES')
kill.head()
kill.name.value_counts()
#MOST COMMON 15 NAME or SURNAME of KILLED PEOPLE

seperate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*seperate)

name_list = a+b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)

#

plt.figure(figsize=(15,10))

ax = sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of Killed People')

plt.ylabel('Frequency')

plt.title('Most Common 15 Name or Surname of Killed People')
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.City.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool['Geographic Area'].value_counts()
# High school graduation rate of the population that is older than 25 in states

percent_over_25_completed_highSchool.percent_completed_hs.replace('-',0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in  area_list : 

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs) / len(x)

    area_highschool.append(area_highschool_rate)

#sorting

data = pd.DataFrame({'area_list': area_list, 'area_highschool_ratio' : area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values

sorted_data2 = data.reindex(new_index)

#visualization

plt.figure(figsize = (15,10))

sns.barplot(x  = sorted_data2['area_list'], y = sorted_data2['area_highschool_ratio'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given States' Population Above 25 that has Graduated High School")
share_race_city.head()
share_race_city.info()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_city.replace('-',0.0,inplace = True)

share_race_city.replace(['(X)'],0.0, inplace = True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share_race_city['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list : 

    x = share_race_city[share_race_city['Geographic area'] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

    

#Visualization

f,ax = plt.subplots(figsize = (15,15))

sns.barplot(x=share_white,y=area_list,color = 'yellow',alpha = 0.5, label = 'White')

sns.barplot(x=share_black,y=area_list,color = 'blue',alpha = 0.7,label = 'African American')

sns.barplot(x=share_native_american, y=area_list, color ='cyan',alpha = 0.6 ,label = 'Native American')

sns.barplot(x=share_asian, y= area_list,color = 'green', alpha = 0.6, label= 'Asian')

sns.barplot(x=share_hispanic,y=area_list,color= 'red',alpha = 0.6, label = 'Hispanic')

ax.legend(loc = 'lower right', frameon = True)

ax.set(xlabel = 'Percentage of Races',ylabel = 'States', title = "Percentage of State's Population According to Races" )

#Point Plot

#Highschool graduation rate  vs Poverty Rate of Each State

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio'] / max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio'] / max(sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis = 1)

data.sort_values('area_poverty_ratio',inplace = True)

#Visulization

f,ax1 = plt.subplots(figsize = (20,10))

sns.pointplot(x = 'area_list', y= 'area_poverty_ratio',data = data,color = 'lime',alpha = 0.9)

sns.pointplot(x = 'area_list', y= 'area_highschool_ratio',data = data, color = 'red',alpha = 0.8)

plt.text(36,0.5,'HighSchool Graduate Ratio',color = 'red', fontsize = 20, style = 'italic')

plt.text(36,0.45,'Poverty Ratio', color = 'lime', fontsize = 20, style = 'italic')

plt.xlabel('States',fontsize = 18, color = 'purple')

plt.ylabel('Values',fontsize = 18, color = 'purple')

plt.title('HighSchool Graduate VS Poverty Rate',fontsize = 25,color = 'blue')

plt.grid()



data.head()
#JOINT PLOT

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation

g = sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind = "kde", size = 7)

plt.savefig('graph.png')
data.head()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data, size = 5, ratio = 3,color = "r")
#PIE CHART

kill.race.head(10)
kill.race.value_counts()
#Race Rate According in Kill Data

kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values

#Visualization

plt.figure(figsize= (10,10))

plt.pie(sizes,explode = explode,labels = labels, colors = colors, autopct = '%1.1f%%')

plt.title('Killed People According to Races', color = 'blue',fontsize = 20)
#LM PLOT

data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x = 'area_poverty_ratio',y='area_highschool_ratio',data=data)
#KDE PLOT

data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut = 2)
#VIOLIN PLOT

data.head()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2,rot = -.5, dark = .3)

sns.violinplot(data = data, palette = pal, inner = "points")
#HEATMAP

data.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(),annot=True,linewidths = 0.5,linecolor = "red",fmt  = '.1f',ax=ax)

#BOX PLOT

kill.head()
kill.manner_of_death.unique()
kill.manner_of_death.value_counts()
# Plot the orbital period with horizontal boxes

sns.boxplot(x = 'gender',y = 'age', hue = 'manner_of_death',data = kill, palette = 'PRGn')
#       ** SWARM PLOT **

kill.head()
plt.figure(figsize= (10,10))

sns.swarmplot(x= "gender",y = "age", hue = "manner_of_death",data = kill)

#PAIR PLOT

data.head()
sns.pairplot(data)
#COUNT PLOT

kill.gender.value_counts()
kill.head()
# kill properties

# Manner of death

sns.countplot(kill.gender)

plt.title('gender',color= 'blue',fontsize = 15)
sns.countplot(kill.manner_of_death)

plt.title('manner of death',color= 'blue',fontsize = 15)
#kill weapon

armed = kill.armed.value_counts()

print(armed)

plt.figure(figsize = (10,8))

sns.barplot(x = armed[:7].index, y = armed[:7].values)

plt.xlabel('Weapon Types')

plt.ylabel('Number of Weapon')

plt.title('Kill Weapon',color = 'blue', fontsize = 15)
#Age of Killed People

above25 = ['above25' if i>=25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.title('Age of Killed People',color = 'blue', fontsize = 15)

plt.ylabel('Number of Killed People')
#Race of Killed People

sns.countplot(data = kill, x = 'race')

plt.title('Race of Killed People', color= 'blue', fontsize = 15)
#Most Dangerous Cities

city = kill.city.value_counts()

plt.figure(figsize = (10,7))

sns.barplot(x = city[:12].index, y = city[:12].values)

plt.xticks(rotation = 45)

plt.title('Most Dangerous Cities', color = 'blue', fontsize = 15)
#Most Dangerous States

state = kill.state.value_counts()

plt.figure(figsize = (10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most Dangerous States',color = 'blue', fontsize = 15)
#Having Mental Illness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental Illness')

plt.ylabel('Number of Mental Illness')

plt.title('Having Mental Illness or not', color = 'blue', fontsize = 15)
#Threat Types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat Types', color = 'blue', fontsize =15)
#Flee Types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee Types', color = 'red' ,fontsize = 15 ) 
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'red',fontsize = 15)
#Kill numbers from States in Kill Data

sta = kill.state.value_counts().index[:10]

sns.barplot(x = sta, y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers of States', color = 'red', fontsize = 15)