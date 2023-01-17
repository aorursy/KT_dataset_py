import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
#percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level.info()
area_list = percentage_people_below_poverty_level["Geographic Area"].unique()
area_list
poverty_ratio=[]
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i]
    area_poverty_rate = sum(x.poverty_rate) / len(x)
    poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area list':area_list,'poverty ratio':poverty_ratio})
new_index = (data['poverty ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
    
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area list'], y=sorted_data['poverty ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate of Each States')
kill.head()
#kill["name"].value_counts()
seperate = kill["name"][kill["name"] != 'TK TK'].str.split()
a,b = zip(*seperate)
name_list = a+b
name_count = Counter(name_list)
name_count = name_count.most_common(15)
x,y = zip(*name_count)
x,y = list(x),list(y)

plt.figure(figsize=(10,8))
ax = sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))
plt.xlabel("Name or Surname Of Killed People")
plt.ylabel("Freq")
plt.title("Most common 15 Name or Surname of killed people")
percent_over_25_completed_highSchool.head()
#percent_over_25_completed_highSchool["percent_completed_hs"].value_counts()
percent_over_25_completed_highSchool.replace(["-"],0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
percent_over_25_completed_highSchool.info()
arealist = list(percent_over_25_completed_highSchool['Geographic Area'].unique())
area_highSchool = []
for i in arealist:
    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"] == i]
    highSchool_rate = sum(x.percent_completed_hs)/len(x)
    area_highSchool.append(highSchool_rate)
data = pd.DataFrame({'HighSchool Area':area_list,'Ratio':area_highSchool})
newIndex = data['Ratio'].sort_values(ascending= False).index.values
sorted_list = data.reindex(newIndex)
plt.figure(figsize=(10,8))
sns.barplot(sorted_list["HighSchool Area"],sorted_list["Ratio"],palette=sns.cubehelix_palette(len(area_highSchool)))
plt.xlabel("High School Area")
plt.ylabel("Ratio")
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
plt.xticks(rotation = 90)
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

sorted_list.head(2)
sorted_data.head()
sorted_list["Ratio"] = sorted_list["Ratio"] / max(sorted_list["Ratio"])
sorted_data["poverty ratio"] = sorted_data["poverty ratio"]/ max(sorted_data["poverty ratio"])
data = pd.concat([sorted_data,sorted_list["Ratio"]],axis=1)
data.sort_values('poverty ratio',inplace=True)
data.head()

sorted_list.head()
f,ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x= 'area list',y ='poverty ratio',data=data,color='red')
sns.pointplot(x= 'area list',y = 'Ratio', data = data,color='blue')
plt.xlabel("States",fontsize = 15)
plt.ylabel("Values",fontsize = 15)
plt.title("Poverty Ratio (Red) vs High School Gradate (Blue) ",fontsize = 20)
plt.grid()
from scipy import stats
g = sns.jointplot(data["poverty ratio"],data["Ratio"],kind='kde',size = 7)
g = g.annotate(stats.pearsonr)
#pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
plt.show()
g = sns.jointplot("poverty ratio","Ratio",data = data,color = 'r')

kill.race.head(10)
kill.race.unique()
kill.race.dropna(inplace=True)
label = kill.race.value_counts().index
size = kill.race.value_counts().values
explode = [0,0,0,0,0,0]
color = ["red","blue","green","yellow","black"]

plt.figure(figsize=(10,10))
plt.pie(size,explode = explode,labels = label,colors=color,autopct='%1.1f%%')
plt.title("Killed People According to Races",color = "black",fontsize = 10)
sns.lmplot(data=data,x="poverty ratio",y="Ratio")
plt.show()
data.head()
sns.kdeplot(data["poverty ratio"],data["Ratio"], shade=True, cut=3)
plt.show()
data.corr()
f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(data.corr(),annot=True,ax=ax)
plt.show()
#Manner of death
kill.manner_of_death.unique()
sns.boxplot(x='gender',y="age",hue="manner_of_death",data=kill)
plt.show()
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)
plt.show()
sns.pairplot(data)
plt.show()
# kill properties
# Manner of death
sns.countplot(kill.gender)
plt.title("Kill table according to gender")
armed = kill.armed.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(armed[:7].index,armed[:7].values)
plt.xlabel("Weapons")
plt.ylabel("Number of Weapon")
plt.title("Kill Weapon",fontsize = 15, color ="black")
above25 = ['above25' if i >25 else 'below25' for i in kill.age]
df = pd.DataFrame({'age':above25})
sns.countplot(df.age)
plt.title("Age of killed people")
plt.xlabel("Age")
plt.ylabel("Number of")
sns.countplot(kill.race)
plt.title("Race of Killed People")
city = kill.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(city[:11].index,city[:11].values)
plt.title("Most dangerous cities",color="black")
plt.show()

state = kill.state.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(state[:20].index,state[:20].values)
plt.title("Most Dangerous States",color="black")
plt.show()
sns.countplot(kill.signs_of_mental_illness)
plt.ylabel("Number of")
plt.xlabel("Mental Illness")
plt.title("Having mental illness or not",fontsize=10)
plt.show()
kill.columns
sns.countplot(kill.threat_level)
plt.xlabel("Threat Types")
plt.ylabel("Number of")
plt.title("Threat Types")
plt.show()