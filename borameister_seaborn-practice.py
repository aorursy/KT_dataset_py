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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True) # exchange ' - ' with '0.0'
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float) # turn into float
area_list = percentage_people_below_poverty_level['Geographic Area'].unique() 
area_poverty_ratio = []

for i in  area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area'] == i] # for every state
    area_poverty_ratio.append(sum(x.poverty_rate)/len(x)) # ratio 
    
data = pd.DataFrame({'area_list':area_list,'poverty_rate':area_poverty_ratio}) # new dataframe thru dictionary
newindex = (data.poverty_rate.sort_values(ascending=False)).index.values   #take the index of the sorted data
sorted = data.reindex(newindex)  # set the new index to 'sorted'


plt.figure(figsize=(15,10))                                  # figsize with matplotlib
sns.barplot(data = sorted, x='area_list',y='poverty_rate')   # barplot 
plt.xticks(rotation=30)                                      # rotation of the states on x axis
plt.xlabel('Area List')                                      
plt.ylabel('Poverty Ratio')                                 
plt.title("Powerty Rate with Respect to States")
plt.show()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True) # replace ' - ' with 0.0
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float) # float for calculation


area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique()) # area list
area_highschool = []

for i in area_list:
    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i] # filtered dataframe: x
    area_highschool.append(sum(x.percent_completed_hs)/len(x))              

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool}) #new dataframe
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values # sort and take the index
sorted2 = data.reindex(new_index)  

# visualization
plt.figure(figsize=(15,10)) # figsize thru matplotlib.pyplot
sns.barplot(x='area_list', y= 'area_highschool_ratio', data=sorted2) # barplot
plt.xticks(rotation= 45) # rotation of the values on x axis
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated From High School")
plt.show()
kill.name.value_counts()
reserved = kill.name[kill.name != 'TK TK'].str.split() # take all names expect for 'TK TK'
a,b = zip(*reserved)                                    # unzip a,b (in Tuple format)
name_list = a+b
most_common = Counter(name_list).most_common(15) # count and find the most common 15 names or surnames

x,y = zip(*most_common)
x,y = list(x),list(y)
most_commonnames = pd.DataFrame({"names":x,"count":y})

### visualization
plt.figure(figsize=(10,10))
sns.barplot( x=most_commonnames['names'], y=most_commonnames['count'],palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Names or Surnames of Killed People')
plt.ylabel('Frequency')
plt.show()
most_common

share_race_city.info() # we have 4 different races, and the objects written numerically must be turned into float for calculation.
share_race_city.head()
#share_race_city.share_white.value_counts()
#share_race_city.share_black.value_counts()
#share_race_city.share_native_american.value_counts()
#share_race_city.share_asian.value_counts()
#share_race_city.share_hispanic.value_counts()
share_race_city.replace(['-'],0.0,inplace=True) # replace '-' and save it (inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
share_race_city.share_white = share_race_city.share_white.astype(float)
share_race_city.share_black = share_race_city.share_black.astype(float)
share_race_city.share_native_american = share_race_city.share_native_american.astype(float)
share_race_city.share_asian = share_race_city.share_asian.astype(float)
share_race_city.share_hispanic = share_race_city.share_hispanic.astype(float)

share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []
area_list = list(share_race_city['Geographic area'].unique())

for each in area_list:
    x= share_race_city[share_race_city['Geographic area'] == each]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))
    
plt.subplots (figsize=(8,11))
sns.barplot(x=share_white, y =area_list ,color='yellow',alpha=0.6, label='White')
sns.barplot(x=share_black, y =area_list, color='purple',alpha=0.6, label='Black')
sns.barplot(x=share_native_american, y =area_list , color='green', alpha=0.5, label='Native American')
sns.barplot(x=share_asian, y =area_list , color='brown',alpha=1,label='Asian')
sns.barplot(x=share_hispanic, y =area_list , color='red',alpha=0.8,label='Hispanic')
plt.title('Percentage of Population According to Races')
plt.xlabel('Areas')
plt.ylabel('Percentage of Races')
plt.legend(loc='upper right',frameon = True)
plt.show()
sorted.head()
sorted2.head()
data
sorted['poverty_rate']= sorted['poverty_rate']/max(sorted['poverty_rate'])                      # normalization for graph scale
sorted2['area_highschool_ratio']= sorted2['area_highschool_ratio']/max(sorted2['area_highschool_ratio'])  #same here
data = pd.concat([sorted,sorted2['area_highschool_ratio']],axis=1)
data.sort_values('poverty_rate',inplace=True)

plt.subplots(figsize=(13,8))
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red')
sns.pointplot(x='area_list',y='poverty_rate',data=data,color='black')
plt.xlabel('States',color='magenta',fontsize=15)
plt.ylabel('Rates',color='magenta', fontsize=15)
plt.title('Poverty Rate vs High School Graduation Rate',color='yellow',fontsize=20)
plt.grid()
plt.show()

plt.figure(figsize=(15,15))
sns.jointplot(x=data.area_highschool_ratio,y=data.poverty_rate,kind='kde',height=7) # height determines the figsize (it will be square)
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# scatter = scatter plot,  reg = lm plot, kde = kde plot
plt.xlabel('High School',fontsize=15)
plt.ylabel('Poverty',fontsize=15)
plt.show()
kill.race.unique()
kill.race.value_counts()
kill.race.dropna(inplace=True)
labels = kill.race.value_counts().index
slices = kill.race.value_counts().values
colors = ["pink","red","blue","green","cyan","orange"]
explode = [0,0,0,0,0,0]

plt.figure(figsize=(5,5))
plt.pie(slices,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.show()
data.head()
sns.lmplot(data=data, x='area_highschool_ratio',y='poverty_rate')
plt.show()
data.head()
plt.figure(figsize=(8,8))
sns.kdeplot(data.poverty_rate, data.area_highschool_ratio, shade=True, cut=4) # first written : x axis, second written : y axis
plt.show()
plt.figure(figsize=(5,5))
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)  # copied and pasted for colorization
sns.violinplot(data=data,palette=pal,inner="point")
plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5,linecolor="red", fmt= '.1f')
plt.show() 
# annot: values on the squares.
# fmt: int or float
# ax = 
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn") 
plt.show()
plt.figure(figsize=(10,10))
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill,size=15) # size : size of the cells
plt.show()
# Shows "BAR PLOT" & "SCATTER PLOT side by side"
sns.pairplot(data)
plt.show()
plt.figure(figsize=(20,10))
sns.countplot(kill.armed)
plt.title("gender",color = 'red',fontsize=15)
armed = kill.armed.value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=armed[:8].index,    y=armed[:8].values)  # value_count : index , values
plt.ylabel('Number of Weapon')
plt.xlabel('Weapon Types')
plt.title('Kill weapon',color = 'blue',fontsize=15)
