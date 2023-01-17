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
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_high_school = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
police_killing_us = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
share_race_by_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level['Geographic Area'].unique()
#Bar Plot

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending = True)).index.values
sorted_data = data.reindex(new_index)


#Visualization

plt.figure(figsize=(25,20))
sns.barplot(x = sorted_data['area_list'], y = sorted_data['area_poverty_ratio'])
plt.xticks(rotation = 0)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')














police_killing_us.head()
#police_killing_us.value_counts()
['ali','haydar']

police_killing_us.info()
police_killing_us.name.value_counts()
#Most common 15 Name or Surname of killed People

separate = police_killing_us.name[police_killing_us.name != 'TK TK'].str.split()
a,b = zip(*separate)
name_list = a + b
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
x,y = zip(*most_common_names)
x,y = list(x),list(y)

#Visualization

plt.figure(figsize = (25,20))
sns.barplot(x = x, y = y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed People')
plt.ylabel('Frequency')
plt.title('Most Common 15 Name or Surname of killed People')



percent_over_25_completed_high_school.head()
percent_over_25_completed_high_school.info()

percent_over_25_completed_high_school.percent_completed_hs.value_counts()
percent_over_25_completed_high_school.percent_completed_hs.replace(['-'],0.0, inplace = True)
percent_over_25_completed_high_school.percent_completed_hs = percent_over_25_completed_high_school.percent_completed_hs.astype(float)
area_list = list(percent_over_25_completed_high_school['Geographic Area'].unique())
area_high_school = []
for i in area_list:
    x = percent_over_25_completed_high_school[percent_over_25_completed_high_school['Geographic Area']==i]
    area_high_school_rate = sum(x.percent_completed_hs)/len(x)
    area_high_school.append(area_high_school_rate)
#Sorting
data = pd.DataFrame({'area_list': area_list,'area_high_school_ratio':area_high_school})
new_index = (data['area_high_school_ratio'].sort_values(ascending = False)).index.values
sorted_data2 = data.reindex(new_index)
#Visualization
plt.figure(figsize = (25,20))
sns.barplot(x = sorted_data2['area_list'], y = sorted_data2['area_high_school_ratio'])
plt.xticks(rotation = 45)
plt.xlabel('Area List')
plt.ylabel('Area High School Rate')
plt.title('Percantage of Given State`s Population Above 25 that has Graduated High School')

share_race_by_city.head()
share_race_by_city.tail()
share_race_by_city.replace(['-'],0.0, inplace = True)
share_race_by_city.replace(['(X)'],0.0, inplace = True)
share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_by_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []
for i in area_list:
    x = share_race_by_city[share_race_by_city['Geographic area']==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))
    
#Visualization 
f, ax = plt.subplots(figsize = (25,20))
sns.barplot(x = share_white, y = area_list, color = 'green', alpha = 0.5, label = 'White')
sns.barplot(x = share_black, y = area_list, color = 'black', alpha = 0.5, label = 'Black')
sns.barplot(x = share_native_american, y = area_list, color = 'pink', alpha = 0.5, label = 'Native American')
sns.barplot(x = share_asian, y = area_list, color = 'blue', alpha = 0.5, label = 'Asian')
sns.barplot(x = share_hispanic, y = area_list, color = 'brown', alpha = 0.5, label = 'Hispanic')

ax.legend(loc = 'lower light', frameon = True)
ax.set(xlabel = 'Percentage of Races', ylabel = 'States', title = 'Percentage of State`s Population According to Races')
    
    
    
    
    
    

percentage_people_below_poverty_level.head()
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
sorted_data2['area_high_school_ratio'] = sorted_data2['area_high_school_ratio']/max(sorted_data2['area_high_school_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_high_school_ratio']],axis = 1)
data.sort_values('area_poverty_ratio',inplace = True)
#Visualization
f, ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x = 'area_list', y = 'area_poverty_ratio', data = data, color = 'lime', alpha = 0.8)
sns.pointplot(x = 'area_list', y = 'area_high_school_ratio', data = data, color = 'red', alpha = 0.8)
plt.text(40,0.6,'high school graduate ratio', color = 'red', fontsize = 17, style = 'italic')
plt.text(40,0.55,'Poverty Ratio', color = 'lime', fontsize = 18, style = 'italic')
plt.xlabel('States', fontsize = 15, color = 'blue')
plt.ylabel('High School Graduate Vs Poverty Rate', fontsize = 20, color = 'blue')
plt.title('High School Graduate VS Poverty Rate', fontsize = 20, color = 'blue')
plt.grid()





data.head()
#Joint Plot
g = sns.jointplot(data.area_poverty_ratio, data.area_high_school_ratio, kind = 'kde', size = 17, color = 'red')
plt.savefig('graph.png')
plt.show()
#Joint Plot
g = sns.jointplot(data.area_poverty_ratio, data.area_high_school_ratio, size = 14,ratio =3 , color = 'red')
plt.savefig('graph.png')
plt.show()
#Pie Chart

police_killing_us.head()




police_killing_us.race.dropna(inplace = True)
labels = police_killing_us.race.value_counts().index
colors = ['pink','black','brown','yellow','red','gray']
explode = [0,0,0,0,0,0]
sizes = police_killing_us.race.value_counts().values

#Visualization

plt.figure(figsize = (20,15))
plt.pie(sizes,explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%') 
plt.title('Killed People According to Races',color = 'blue', fontsize = 15)


#LM Plot

sns.lmplot(x = "area_poverty_ratio", y = "area_high_school_ratio",data = data)
plt.show



#Kde Plot
sns.kdeplot(data.area_poverty_ratio, data.area_high_school_ratio,shade = True , cut = 5, color = 'black')
plt.show

#Violin Plot
pal = sns.cubehelix_palette(2, rot = -.5, dark = -3)
sns.violinplot(data = data, palette = pal, inner = 'points')
plt.show()
data.corr()
#Heat Map this one for using correlation


f , ax = plt.subplots(figsize =(5,5))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.1f',ax=ax)
plt.show()

#Box Plot
sns.boxplot(x = "gender", y = "age", hue = "manner_of_death", data = police_killing_us, palette = "PRGn") 
police_killing_us.head()
#Swarm Plot

sns.swarmplot(x = "gender", y = "age", hue = "manner_of_death", data = police_killing_us) 
plt.show()
#Pair Plot
sns.pairplot(data)
plt.show
#Count Plot
police_killing_us.manner_of_death.value_counts()
sns.countplot(police_killing_us.gender)
plt.title("gender", color = 'blue', fontsize = 15)
armed = police_killing_us.armed.value_counts()
plt.figure(figsize = (10,7))
sns.barplot(x = armed[:7].index, y = armed[:7].values)
plt.xlabel("Weapon Types")
plt.ylabel("Number")
plt.title('Kill Weapon', size = 15 , color = 'blue')
above25 = ['above25' if i>= 25 else 'below25' for i in police_killing_us.age] 
df = pd.DataFrame({'age':above25})
sns.countplot (x = df.age)
plt.ylabel('Number of Killed People')
plt.title('Age of killed people', color = 'blue', fontsize = 15)
sns.countplot(data = police_killing_us, x ='race')
plt.title('Race of killed People', color = 'blue', fontsize = 17)
city = police_killing_us.city.value_counts()
plt.figure(figsize = (10,7))
sns.barplot(x = city[:12].index, y = city[:12].values,)
plt.xticks(rotation = 90)
plt.title('Most dangerous cities', color = 'blue', fontsize = 15)
state = police_killing_us.state.value_counts()
plt.figure(figsize = (10,7))
sns.barplot(x = state[:10].index, y = state[:10].values)
plt.title('Most dangerous state in US')
#Having Mental Ilness or not for killed People
sns.countplot(police_killing_us.signs_of_mental_illness)
plt.xlabel('Mental Illness')
plt.ylabel('Number of mental illness')
plt.title('Having mental illness or not ', color = 'blue', fontsize = 17)





#Threat Types
sns.countplot(police_killing_us.threat_level)
plt.xlabel('Threat Types')
plt.title('Threat Types', color = 'blue', fontsize = 17)







sns.countplot(police_killing_us.flee)
plt.xlabel('Flee Types')
plt.title('Flee Types', color = 'blue', fontsize = 17)
sns.countplot(police_killing_us.body_camera)
plt.xlabel('Having Body Camera')
plt.title('Having Body Camera', color = 'blue', fontsize = 17)

sta = police_killing_us.state.value_counts().index[:10]
sns.barplot(x = sta, y = police_killing_us.state.value_counts().values[:10])
plt.title('Kill Numbers from States', color = 'blue', fontsize = 17 )
