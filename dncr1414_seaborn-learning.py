# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
median_house_hold_in_come.head()
percentage_people_below_poverty_level.head()
percent_over_25_completed_highSchool.head()
share_race_city.head()
kill.head()
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

areaList=list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio=[]

for i in areaList:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

    

data=pd.DataFrame({'areaList': areaList,'area_poverty_ratio':area_poverty_ratio})

new_index=(data['area_poverty_ratio'].sort_values(ascending=True)).index.values

sorted_data = data.reindex(new_index)



#Visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['areaList'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 60)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
share_race_city.head()
share_race_city.info()
#share_race_city.share_white.value_counts()

#share_race_city.share_black.value_counts()

#share_race_city.share_native_american.value_counts()

#share_race_city.share_asian.value_counts()

#share_race_city.share_hispanic.value_counts()

share_race_city.replace(['-'],0.0,inplace=True)

share_race_city.replace(['(X)'],0.0,inplace=True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']]=share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

areaList=list(share_race_city['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]



for i in areaList:

    x = share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

    

f,ax=plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=areaList,color='green',alpha=0.5,label='White')

sns.barplot(x=share_black,y=areaList,color='black',alpha=0.5,label='African American')

sns.barplot(x=share_native_american,y=areaList,color='red',alpha=0.5,label ='Native American')

sns.barplot(x=share_asian,y=areaList,color='yellow',alpha=0.5,label='Asian')

sns.barplot(x=share_hispanic,y=areaList,color='pink',alpha=0.6,label='Hispanic')

    

ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

areaList = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in areaList:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

# sorting

data = pd.DataFrame({'areaList': areaList,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['areaList'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
sorted_data.head()

sorted_data2.head()
sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)



#Visualize

f,x1=plt.subplots(figsize=(20,10))

sns.pointplot(x='areaList',y='area_poverty_ratio',data=data, color='green')

sns.pointplot(x='areaList',y='area_highschool_ratio',data=data, color='black')

plt.text(40,0.4,'high school graduate ratio',color='black',fontsize = 17)

plt.text(40,0.35,'poverty ratio',color='green',fontsize = 18)

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()

data.head()
g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind='kde' ,height=7)



#pearsonr

from scipy import stats

g = g.annotate(stats.pearsonr)



plt.show()
t=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,ratio=6,color="orange")
kill.race.head(10)
kill.race.dropna(inplace=True)

labels=kill.race.value_counts().index

kill.race.value_counts()

colors=['grey','orange','blue','red','green','pink']

explode=[0,0,0,0,0,0]

sizes=kill.race.value_counts().values



#visualite

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
data.head()
sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio" ,data=data)
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio, shade=True, cut=5)

plt.show()
pal=sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data,palette=pal,inner="points")

plt.show()
data.head()
data.corr()
f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidths=0.5 ,ax=ax)

plt.show()
kill.head()
sns.boxplot(x="gender",y="age",hue="manner_of_death",data=kill)

plt.show()
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)

plt.show()
sns.pairplot(data)

plt.show()
sns.countplot(kill.gender)

sns.countplot(kill.manner_of_death)

plt.title("manner of death", color="DarkBlue")

plt.show()