# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read datas
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()#show us the states of USA
#len (percentage_people_below_poverty_level['Geographic Area'].unique())#show us the states of USA
# Poverty rate of each state
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)#poverty_rate can not be object
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]#take one state but all rows
    area_poverty_rate = sum(x.poverty_rate)/len(x)#x state avarage of poverty
    area_poverty_ratio.append(area_poverty_rate)#add avarage the series
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})#new dataframe list two column
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values#change index by poverty ratio sequence 
#print (type(new_index)) this is an array
#print (new_index)
sorted_data = data.reindex(new_index)#reindex command is regenarate index by new_index
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
kill.info()
kill.head()
kill.name.value_counts()
seperate= kill.name[kill.name !='TK TK'].str.split()
#print(seperate) #like an array :  [Tim, Elliot]
a,b=zip(*seperate)#zip is use to transform array to tuple 
#print(a,b) --('Tim', 'Lewis', 'John',.... ) name and surname
name_list=a+b
#print (name_list)
name_count = Counter(name_list) 
#print(name_count)
most_common_names = name_count.most_common(15) #most used 15 names and numbers in sheet
x,y = zip(*most_common_names)#x replace of name y is taking numbers
#print (x,y)
x,y = list(x),list(y)#tupbles to array transform
#print (x,y)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))#different diagram types cubehelix_palette by seaborn
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')


percent_over_25_completed_highSchool.head()

percent_over_25_completed_highSchool.info()
x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']=='AL']
x
# High school graduation rate of the population that is older than 25 in states
percent_over_25_completed_highSchool.percent_completed_hs.replace('-',0.0,inplace=True)
percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list =list (percent_over_25_completed_highSchool['Geographic Area'].unique())
area_highschool=[]
for i in area_list:
    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]
    area_highschool_rate=sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)
data=pd.DataFrame({'area_list':area_list,'area_highschool_ratio': area_highschool})
new_index=data.area_highschool_ratio.sort_values(ascending=True).index.values
#print(new_index)
sorted_data2=data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
    
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

# visualization
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )
sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')
sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')
sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')
sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
plt.show()
# high school graduation rate vs Poverty rate of each state
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)
data.sort_values('area_poverty_ratio',inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
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
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", height=7)
plt.savefig('graph.png')#save the figure kaggle's page. 

plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,height=5, ratio=3, color="r")
#combine tow gragh with plot_joint. n_levels is number of line in the graph.
g = (sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,height=5, ratio=3, color="r").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="reg", height=7)
plt.show()
kill.head()
kill.race.value_counts()
#pie chart is member of matplot libraray
kill.race.dropna(inplace=True)#delete nan value rows
labels=kill.race.value_counts().index#assign of grouped value of race in index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0.1,0.1,0.1,0.1,0.1,0.1] #throw the slice
sizes = kill.race.value_counts().values#value count of numbers in to sizes 

plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races',color='black',fontsize=15)
# kdeplot
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3)#cut is size of shape in the graph, shade is fill in shape with color
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)#it is like cumulation of each part on the same graph
sns.violinplot(data=data, palette=pal, inner="points")#use data dataframe and pal for palette and inner="points" is point in the shape
plt.show()
#correlation map
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
#annot=True is visualation of numbers in the rectangle
#ax=ax is establish the heatmap in to the generated plot. 
plt.show()
kill.head()
kill.head()

a=kill.groupby(['state','race'], as_index=False).agg({"id": "count"})
a=a.sort_values('id', ascending=False)

a.head()
#Killed people and number of races state by state
#for example most people killed in CA and their races is Hispanic, second is white
f,ax1 = plt.subplots(figsize =(20,10))
sns.swarmplot(x="state", y="id",hue="race", data=a)
plt.show()
# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='state',y='id',data=a,color='lime',alpha=0.8)
plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('COUNTS',fontsize = 15,color='blue')
plt.title('Killed People - States',fontsize = 20,color='blue')
plt.grid()
#x is gender, y is age and distirbution about manner of death. we use as data 'kill' and colouring palette 'PRGn'=Purple-Green
sns.boxplot(x='gender', y='age',hue='manner_of_death', data=kill, palette='PRGn')
plt.show()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)
plt.show()
#Pairplot shows us to 4 chart and 2 is point others line plot. And x  and y values are changing.
sns.pairplot(data)
plt.show()
sns.countplot(kill.gender)
#sns.countplot(kill.manner_of_death)
plt.title("gender",color = 'blue',fontsize=15)
sns.countplot(kill.manner_of_death)
plt.title("manner of death" , color='blue', fontsize=15)
armed=kill.armed.value_counts()#armed occur from two colomns that index is name of arm and values is number of used.
#print(armed)
plt.figure(figsize=(10,7))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.xlabel('Weapon Types')
plt.ylabel('Weapon Numbers')
plt.title('Kill weapon',color = 'blue',fontsize=15)
plt.show()
above25=['above25' if i>25 else 'below25' for i in kill.age]#we choose and create a list 
df=pd.DataFrame({'age':above25})
#print (df)
sns.countplot(x=df.age)#we determine the x axis here
plt.ylabel('Number of Killed People')
plt.title('Age of killed people',color = 'blue',fontsize=15)
plt.show()
sns.countplot(data=kill, x='race')#another using of countplot; we can give data and x axis like this <-
plt.title('Race of killed people',color = 'blue',fontsize=15)
plt.show()
kill.head()
#a=kill.set_index(['race','city']).count(level="race")
#kill.city.dropna(inplace=True)
a=kill.sort_values(by=['race'])
a
area_list = list(kill['city'].unique())
share_white1 = []
share_black1 = []
share_native_american1 = []
share_asian1 = []
share_hispanic1 = []
for i in area_list:
    x = kill[kill['city']==i]
# Most dangerous cities
city=kill.city.value_counts()
plt.figure(figsize=(12,10))
sns.barplot(x=city[:12].index, y=city[:12].values)
plt.xticks(rotation=45)
plt.xlabel('Cities')
plt.ylabel('Number of Killed People')
plt.title(' Most dangerous cities',color = 'blue',fontsize=15)
plt.show()


# Most dangerous cities
states=kill.state.value_counts()
plt.figure(figsize=(12,10))
sns.barplot(x=states[:12].index, y=states[:12].values)
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('Number of Killed People')
plt.title(' Most dangerous states',color = 'blue',fontsize=15)
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
# Flee types
sns.countplot(kill.flee)
plt.xlabel('Flee Types')
plt.title('Flee types',color = 'blue', fontsize = 15)




