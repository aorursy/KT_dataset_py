# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
percentage_people_below_poverty_level=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percentage_over_25complete_highschool=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
median_household_income2015=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
police_killingsUs=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")
share_race_bycity=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")


percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()
len(percentage_people_below_poverty_level['Geographic Area'].unique())
#poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list=list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio=[]

for i in area_list:
    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    #area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(sum(x.poverty_rate)/len(x))

data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)

 

#visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
#percentage_people_below_poverty_level.info()
kill.head()

kill.name.value_counts()
#len(kill.name.unique())
len(kill.name.value_counts())
separate=kill.name[kill.name!='TK TK'].str.split()
print(separate)
a,b=zip(*separate)
#print(a)
print(b)
#name_list=a+b
#print(name_list)
#most common 15 name and surname of killed people
separate=kill.name[kill.name!='TK TK'].str.split()
a,b=zip(*separate)
name_list=a+b
#k,l=zip(*name_list)
#print(k)
name_count=Counter(name_list)
#print(name_count)-> counter as list
#print(len(name_list))-> direct number
most_common_names=name_count.most_common(15)
x,y = zip(*most_common_names)
x,y=list(x),list(y)
#print (x)
#print (y)
#
plt.figure(figsize=(15,10))
ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 name or surname killed people')
percentage_over_25complete_highschool.head()
percentage_over_25complete_highschool.info()
percentage_over_25complete_highschool.percent_completed_hs.value_counts()
percentage_over_25complete_highschool.percent_completed_hs.replace(['-'],0.0,inplace=True)
percentage_over_25complete_highschool.percent_completed_hs=percentage_over_25complete_highschool.percent_completed_hs.astype(float)

area_list=list(percentage_over_25complete_highschool['Geographic Area'].unique())
area_highschool=[]

for i in area_list:
    x=percentage_over_25complete_highschool[percentage_over_25complete_highschool['Geographic Area']==i]
    area_highschool.append(sum(x.percent_completed_hs)/len(x))
    

    
#sorting
data=pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})
new_index=(data['area_highschool_ratio'].sort_values(ascending=True).index.values)
sorted_data2=data.reindex(new_index)

#visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Hish School Graduate Rate')
plt.title('Percentage of Given States Population Above 25 that has graduated high school')
share_race_bycity.head()
share_race_bycity.info()
share_race_bycity.replace(['-'],0.0,inplace = True)
share_race_bycity.replace(['(X)'],0.0,inplace = True)
share_race_bycity.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_bycity.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list=list(share_race_bycity['Geographic area'].unique())



share_white=[]
share_black=[]
share_native_american=[]
share_asian=[]
share_hispanic=[]

for i in area_list:
    x=share_race_bycity[share_race_bycity['Geographic area']==i]
    share_white.append(sum(x.share_white) / len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))
    
    
#visualization
f,ax=plt.subplots(figsize=(9,15))
sns.barplot(x=share_white,y=area_list,color='green',alpha=0.5,label='White')
sns.barplot(x=share_black,y=area_list,color='blue',alpha=0.5,label='Black')
sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha=0.5,label='American')
sns.barplot(x=share_asian,y=area_list,color='yellow',alpha=0.5,label='Asian')
sns.barplot(x=share_hispanic,y=area_list,color='red',alpha=0.5,label='Hispanic')

ax.legend(loc='lower right',frameon = True) 
ax.set(xlabel='Percentage of Races', ylabel='States', title="Percentage of State's Population According to Race")



sorted_data.head()
#normalization example
x=[1,2,3,4,5]
y1=[1,2,3,4,5]
y2=[100,101,102,103,104]
import matplotlib.pyplot as plt
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()


# high school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])#-> for normalization

data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1) #->columnlarla birlşetirdik
data.sort_values('area_poverty_ratio', inplace=True)

#visualize
f,ax1=plt.subplots(figsize=(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime', alpha=0.5)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
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

g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind="kde", size=5, ratio=3, color='r')
plt.savefig('graph.png')
plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, data=data, size=5, color='r',ratio=3)
plt.savefig('graph.png')
plt.show()
data.head()
kill.race.head(15)
kill.race.value_counts()
kill.race.dropna(inplace=True)
labels=kill.race.value_counts().index
print(labels)
colors=['grey','blue','red','yellow','green','brown']
explode=[0,0,0,0,0,0]
#sizes=kill.race.value_counts()
sizes=kill.race.value_counts().values
print(sizes)

#visual
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races', color='blue', fontsize=15)

data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)
plt.show()
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# cubehelix plot
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=1)
plt.show()
data.head()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()
data.corr()
#correlation map
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
kill.head()
kill.manner_of_death.unique()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
# Plot the orbital period with horizontal boxes
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")
plt.show()
kill.head()
# swarm plot
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)
plt.show()
data.head()
data.head()
# pair plot
sns.pairplot(data)
plt.show()
kill.gender.value_counts()
kill.gender.value_counts().values
kill.head()
# kill properties
# Manner of death
sns.countplot(kill.gender)
#sns.countplot(kill.manner_of_death)
plt.title("gender",color = 'blue',fontsize=15)
# kill weapon
armed = kill.armed.value_counts()
print(armed)
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
# Most dangerous cities
city = kill.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:12].index,y=city[:12].values)
plt.xticks(rotation=45)
plt.title('Most dangerous cities',color = 'blue',fontsize=15)
# most dangerous states
state = kill.state.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=state[:20].index,y=state[:20].values)
plt.title('Most dangerous state',color = 'blue',fontsize=15)
# Having mental ilness or not for killed people
sns.countplot(kill.signs_of_mental_illness)
plt.xlabel('Mental illness')
plt.ylabel('Number of Mental illness')
plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
# Threat types
sns.countplot(kill.threat_level)
plt.xlabel('Threat Types')
plt.title('Threat types',color = 'blue', fontsize = 15)
# Flee types
sns.countplot(kill.flee)
plt.xlabel('Flee Types')
plt.title('Flee types',color = 'blue', fontsize = 15)
# Having body cameras or not for police
sns.countplot(kill.body_camera)
plt.xlabel('Having Body Cameras')
plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)
# Kill numbers from states in kill data
sta = kill.state.value_counts().index[:10]
sns.barplot(x=sta,y = kill.state.value_counts().values[:10])
plt.title('Kill Numbers from States',color = 'blue',fontsize=15)