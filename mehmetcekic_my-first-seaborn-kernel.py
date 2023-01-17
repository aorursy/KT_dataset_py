# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read datas
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool= pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True) # assignin 0.0 to inproper datas
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area'] == i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')

# Most common 15 Name or Surname of killed people

separate = kill.name[kill.name != 'TK TK'].str.split()
a,b = zip(*separate)
name_list = a+b
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
x,y = zip(*most_common_names)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
ax = sns.barplot(x=x, y=y, palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
percentage_ratio = []
for i in area_list:
    data_x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area'] == i]
    ratio = sum(data_x.percent_completed_hs)/len(data_x)
    percentage_ratio.append(ratio)

over_25 = pd.DataFrame({"Geographic_Area":area_list, "High_School_Percentage":percentage_ratio})
new_index = (over_25.High_School_Percentage.sort_values(ascending=True)).index.values
sorted_over_25 = over_25.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_over_25.Geographic_Area, y=sorted_over_25.High_School_Percentage)
plt.xticks(rotation=45)
plt.title("High school graduation rate of the population that is older than 25 in states")
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')

share_race_city.replace(['-'],0.0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for i in area_list:
    x = share_race_city[share_race_city['Geographic area'] == i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))

f,ax = plt.subplots(figsize=(9,15))
sns.barplot(x=share_white,y=area_list, color ='green', alpha=0.5, label = 'White')
sns.barplot(x=share_black,y=area_list, color ='blue', alpha=0.7, label = 'African American')
sns.barplot(x=share_native_american,y=area_list, color ='cyan', alpha=0.6, label = 'White')
sns.barplot(x=share_asian,y=area_list, color ='yellow', alpha=0.6, label = 'Asian')
sns.barplot(x=share_hispanic,y=area_list, color ='red', alpha=0.6, label = 'Hispanic')

ax.legend(loc='lower right', frameon = True)
ax.set(xlabel='Percentage of Races', ylabel='States', title= "Percentage of State's Population According to Races")

# High school graduation rate vs Poverty rate of each state 

#sorted_over_25.head()
#sorted_data.head()
#Let's normalize the data
sorted_over_25["High_School_Percentage"] = sorted_over_25["High_School_Percentage"] / max(sorted_over_25["High_School_Percentage"])
sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"] / max(sorted_data["area_poverty_ratio"])
data = pd.concat([sorted_data,sorted_over_25["High_School_Percentage"]], axis=1)
data.sort_values('area_poverty_ratio',inplace=True)
#visualize
f,ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='High_School_Percentage',data=data,color='red',alpha=0.8)
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
g = sns.jointplot(data.area_poverty_ratio, data.High_School_Percentage, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot("area_poverty_ratio","High_School_Percentage", data=data, size=5, ratio=3,color="r")
#Race rates according to kill data
#kill.head()
labels = kill.race.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
sizes = kill.race.value_counts().values
explode = [0,0,0,0,0,0]

plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels = labels, colors= colors, autopct='%1.1f%%')
plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="area_poverty_ratio", y="High_School_Percentage", data=data)
plt.show()
sns.kdeplot(data.area_poverty_ratio, data.High_School_Percentage, shade = True, cut=5)
plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()

#correlation map
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()

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
sns.pairplot(data)
plt.show()
# kill properties
# Manner of death
sns.countplot(kill.gender)
#sns.countplot(kill.manner_of_death)
plt.title("gender",color = 'blue',fontsize=15)
kill.head()
sns.countplot(kill.manner_of_death)
plt.title("Manner of Death", color='blue',fontsize=15)
armed = kill.armed.value_counts()
plt.figure(figsize=(15,10))
sns.barplot(x=armed[:7].index, y=armed[:7].values)
plt.title("Gun Counts")
plt.xlabel('Armed')
plt.ylabel('Count')
def slice(age):
    if age > 40:
        return 'above 40'
    elif age>18:
        return 'between 18-40'
    else:
        return 'under 18'
    
age_slice = kill.age.apply(slice)
df = pd.DataFrame({'Age':age_slice})
sns.countplot(df.Age)
dangerous = kill.city.value_counts()
plt.figure(figsize=(15,10))
sns.barplot(x=dangerous[:15].index,y=dangerous[:15].values)
plt.title("Most Dangerous City", color="red")
plt.xticks(rotation=45)
plt.xlabel("City")
plt.ylabel("Kill Count")