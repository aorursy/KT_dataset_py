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
#Reading Data

median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

kill= pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.head(10)
percentage_people_below_poverty_level['Geographic Area'].unique()

#Unique geographic areas in dataset
percentage_people_below_poverty_level.poverty_rate.value_counts()
# We ara going to ignore that 197 dashes
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True) #inplace true means after the replace save the new variables over the old one

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float) 

#percent_completed_hs(poverty rate) is still an object(string) we need values so lets transform them to float

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())  #we need to take each state

area_poverty_ratio = [] # empty list

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

# we found the mean of the level and append to the "area_poverty_ratio"

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values   #"ascending=False" =downward

sorted_data = data.reindex(new_index)

# created a new dataframe 
# visualization (Bar Plot)

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
kill.name.value_counts()
# Most common 15 Name or Surname of killed people

separate = kill.name[kill.name != 'TK TK'].str.split()   #we take the non TK names and we slipt the name and the surnames

a,b = zip(*separate)   #split the names and surnames                 

name_list = a+b     #we gather them in to a tuple

name_count = Counter(name_list)     #count the names and surnames    

most_common_names = name_count.most_common(15)  #take most common ones

x,y = zip(*most_common_names)

x,y = list(x),list(y)
#Visualization(Bar Plot)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float) #string>float

area_list=list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []



for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)



#sorting(at the moment data not sorted)



data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)
# visualization (Bar Plot)

plt.figure(figsize=(15,10)) #x=15 y=10

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_city.head()
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
# visualization (Side Bar Plot)

f,ax = plt.subplots(figsize = (12,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='upper right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
#sorted_data= Poverty rate

sorted_data.head()
#sorted_data2= High School Graduation rate

sorted_data2.head()
#High school graduation rate vs Poverty rate of each state

#We have to normalize data first

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

#we unite the two different dataset

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

data.head()
# visualization (Point Plot)

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.5) #(data=data ==> First data means seaborn parameter, Second data means our united data, in this way we tell seaborn which data we are using. )

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.5)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')   #(x=40  y=0,6 Locations)

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", height=10)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,height=5, ratio=3, color="r",kind="scatter")
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,height=5, ratio=3, color="r", kind="reg")
#shows residuals

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,height=5, ratio=3, color="r", kind="resid")
kill.head()
kill.race.head(12)
kill.race.value_counts()
# Race rates according in kill data 

kill.race.dropna(inplace=True) #if there are any empty or NaN value drop them

labels = kill.race.value_counts().index   #labels=index

colors = ['grey','blue','red','yellow','green','brown'] # pie chart colours

explode = [0,0,0,0,0,0] # empty list

sizes = kill.race.value_counts().values  # take the values
# visualization

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')   #autopct='%1.1f%%= one decimal point

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3)

plt.show()
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=False, cut=3)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3) #colour

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
data.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)  #annot=True means values in the heat map # linewidths=0.5 means line thickness between colours

plt.show()
kill.head()
kill.manner_of_death.unique()
kill.gender.unique()
# manner of death : shot or shot and Tasered

# Plot the orbital period with horizontal boxes

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn") #hue means classes

plt.show()
kill.head()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()
data.head()
# pair plot

sns.pairplot(data)

plt.show()
kill.head()
kill.manner_of_death.value_counts()
# kill properties

# Manner of death

sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)
# kill weapon

armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people

above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)