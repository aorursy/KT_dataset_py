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
data_median=pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

data_percentage=pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

data_percentover=pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

data_police_kill=pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")

data_share=pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
data_percentage.head()
data_percentage.info()
#poverty rate of each state

data_percentage.poverty_rate.replace(['-'],0.0,inplace=True)

data_percentage.poverty_rate=data_percentage.poverty_rate.astype(float)

area_list=list(data_percentage['Geographic Area'].unique())

area_poverty_ratio=[]

for i in area_list:

    x=data_percentage[data_percentage['Geographic Area']==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

    

data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})

new_index=(data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=90)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
#most common 15 name or surname of killed people

data_police_kill.head()
data_police_kill.name.value_counts()
separate=data_police_kill.name[data_police_kill.name!='TK TK'].str.split()

a,b=zip(*separate)

name_list=a+b

name_count=Counter(name_list)

most_common_names=name_count.most_common(15)

x,y=zip(*most_common_names)

x,y=list(x),list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most Common 15 Name of Killed People')



data_percentover.head()
data_percentover.info()
data_percentover.percent_completed_hs.value_counts()
data_percentover.percent_completed_hs.replace(['-'],0.0,inplace=True)

data_percentover.percent_completed_hs=data_percentover.percent_completed_hs.astype(float)

area_list=list(data_percentover['Geographic Area'].unique())

area_highschool=[]

for i in area_list:

    x=data_percentover[data_percentover['Geographic Area']==i]

    area_highschool_rate=sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

data=pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})

new_index=(data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)

plt.figure(figsize=(15,10))

ax=sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xlabel('Area List')

plt.ylabel('Area Highschool Ratio')

plt.title('Highscool Graduation Ration in States')
data_share.info()
data_share.head()
#percentage of satate's population

data_share.replace(['-'],0.0,inplace=True)

data_share.replace(['(X)'],0.0,inplace=True)

data_share.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']]=data_share.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list=list(data_share['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]

for i in area_list:

    x=data_share[data_share['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))   

    

f,ax=plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha=0.5,label='White')

sns.barplot(x=share_black,y=area_list,color='red',alpha=0.5,label='Black')

sns.barplot(x=share_native_american,y=area_list,color='blue',alpha=0.5,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='pink',alpha=0.7,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='yellow',alpha=0.5,label='Hispanic')

ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='percentage of races',ylabel='states')
sorted_data.head()
sorted_data2.head()
x=[1,2,3,4,5]

y1=[1,2,3,4,5]

y2=[1000,999,888,777,666]

plt.plot(x,y1)

plt.show
plt.plot(x,y1)

plt.plot(x,y2)

plt.show()
#Highschool graduation rate vs poverty rate each states

sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)



#visualize



f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='orange',alpha=0.8)

plt.text(40,0.6,'highschool graduate ratio',color='orange',fontsize=17,style='italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize=18,style='italic')

plt.xlabel('States',fontsize=15,color='blue')

plt.ylabel('Values',fontsize=15,color='blue')

plt.title('HIGHSCHOOL GRADUATE VS POVERTY RATE',fontsize=20,color='blue')

plt.grid()
data.head()
#Highschool graduation rate vs poverty rate each states with joint plot



g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind='kde',height=7)

plt.savefig('graph.png')

plt.show()



g=sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data,ratio=3,color='r')
data_police_kill.head()
data_police_kill.race.head(15)
#race rates from killed people data

data_police_kill.race.dropna(inplace=True)

labels=data_police_kill.race.value_counts().index

colors=['grey','blue','red','purple','green','pink']

explode=[0,0,0,0,0,0]

sizes=data_police_kill.race.value_counts().values



#visualize



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1%%f')

plt.title('Killed People According to Races',color='blue',fontsize=15)

data.head()
#linear regression within each dataset

sns.lmplot(x='area_poverty_ratio',y='area_highschool_ratio',data=data)

plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut=5)

plt.show()
#numeric values, we can see the most value in the feature with violin plot

pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=data,palette=pal,inner='points')

plt.show()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True, linewidths=.5,fmt='.1f', ax=ax)

plt.show()
#manner of death(shot and tasered)

#gender

#age

#hue=class

sns.boxplot(x='gender',y='age',hue='manner_of_death',data=data_police_kill,palette='PRGn')

plt.show()
#manner of death

#gender

#age

#we can use this plot classification algorithm in machine learning 

sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=data_police_kill)

plt.show()
#this plot visualises all numeric values in data

sns.pairplot(data)

plt.show()
data_police_kill.manner_of_death.value_counts()
sns.countplot(data_police_kill.gender)

sns.countplot(data_police_kill.manner_of_death)

plt.title('manner of death',color='blue',fontsize=15)

plt.show()
armed=data_police_kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Type of Weapon')

plt.title('Kill Weapon',color='blue',fontsize=15)
#age of killed people

above25=['above25' if i>=25 else 'below25' for i in data_police_kill.age]

df=pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of Killed People',color='blue',fontsize=15)

sns.countplot(data=data_police_kill,x='race')

plt.ylabel('Number of Killed People')

plt.xlabel('Races')

plt.title('Race of Killed People')