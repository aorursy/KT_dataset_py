# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come=pd.read_csv("../input/MedianHouseholdIncome2015.csv",encoding="windows-1252")

percentage_people_below_poverty_level=pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

percent_over_25_completed_highSchool=pd.read_csv("../input/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")

share_race_city =pd.read_csv("../input/ShareRaceByCity.csv",encoding="windows-1252")

kill =pd.read_csv("../input/PoliceKillingsUS.csv", encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list=list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_list
area_poverty_ratio=[]

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

#area_poverty_rate

#area_poverty_ratio

data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
data
new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)

sorted_data
plt.figure(figsize=(15,5))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate of Given States')

plt.show()
kill.head()

#kill.name.value_counts()
separate=kill.name[kill.name!='TK TK'].str.split()

a,b=zip(*separate)

name_list=a+b

name_count=Counter(name_list)

most_common_names=name_count.most_common(15)

x,y=zip(*most_common_names)

x,y=list(x),list(y)
plt.figure(figsize=(18,5))

ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of Killed People')

plt.ylabel('Frequency')

plt.title('Most Common 15 Names or Surnames of Killed Peoople')
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.head()
#percent_over_25_completed_highSchool.percent_completed_hs.value_counts()

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

# sorting

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")



sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

f,ax1=plt.subplots(figsize=(15,5))

sns.pointplot(x='area_list', y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list', y='area_highschool_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize=12,style='italic')

plt.text(40,0.6,'poverty ratio',color='lime',fontsize=12,style='italic')

plt.xlabel('States',fontsize=10,color='blue')

plt.ylabel('Values',fontsize=10,color='blue')

plt.title('High School Graduate & Poverty Ratio')

plt.grid()

plt.show()
data.head()
g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind='kde',height=7)

plt.savefig('graph.png')

plt.show()

g=sns.jointplot("area_poverty_ratio","area_highschool_ratio",data=data,size=5,ratio=2,color='r')
kill.race.head()
kill.race.value_counts()
kill.race.dropna(inplace=True)

labels=kill.race.value_counts().index



colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



plt.figure(figsize=(5,5))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='1f.1%%')

plt.title('Killed People According to Races',color='blue',fontsize=12)
sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio",data=data)

plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut=3)

plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data,palette=pal,inner='points')

plt.show()
data.corr()
f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
kill.manner_of_death.unique()
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()
sns.pairplot(data)

plt.show()
kill.gender.value_counts()
sns.countplot(kill.gender)

plt.title("gender",color = 'blue',fontsize=15)
above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})



sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
