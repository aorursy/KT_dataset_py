import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter



%matplotlib inline



import warnings

warnings.filterwarnings('ignore') 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
income=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv',encoding='windows-1252')

highschool=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding='windows-1252')

poverty=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding='windows-1252')

race=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding='windows-1252')

kill=pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding='windows-1252')
poverty.head()
poverty.info()
poverty.poverty_rate.value_counts()
poverty.poverty_rate.replace(['-'],0.0,inplace=True)

poverty.poverty_rate = poverty.poverty_rate.astype(float)

area_list = list(poverty['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = poverty[poverty['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.grid()
kill.tail(20)
separate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*separate)

name_list = a+b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)



plt.figure(figsize=(15,10))

sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most Common 15 Names or Surnames of Killed People")

plt.grid()
kill.loc[kill.name.str.contains(" J. ")]
highschool.percent_completed_hs.value_counts()
highschool.percent_completed_hs.replace(["-"],0.0,inplace=True)

highschool.percent_completed_hs = highschool.percent_completed_hs.astype(float)

area_list = list(highschool["Geographic Area"].unique())

area_hs=[]

for i in area_list:

    x = highschool[highschool["Geographic Area"]==i]

    area_hs_rate = sum(x.percent_completed_hs)/len(x)

    area_hs.append(area_hs_rate)

#sorting 

data=pd.DataFrame({"area_list":area_list,"area_hs_ratio":area_hs})

new_index=(data["area_hs_ratio"].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)

#visualiaztion

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2["area_list"],y=sorted_data2["area_hs_ratio"])

plt.xticks(rotation=45)

plt.xlabel("States")

plt.ylabel("High School Graduate Rate")

plt.title("Percentage of Given State's Population Above 25 that has Graduated High School")

plt.grid()
data.head()
race.head()
race.replace(["-"],0.0,inplace=True)

race.replace(["(X)"],0.0,inplace=True)

race.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]]=race.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

area_list=list(race["Geographic area"].unique())



share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]



for i in area_list:

    x=race[race["Geographic area"]==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

    

#visualization

f,ax=plt.subplots(figsize=(20,15))

sns.barplot(x=share_white,y=area_list,color="green",alpha=0.5,label="White")

sns.barplot(x=share_black,y=area_list,color="blue",alpha=0.7,label="African American")

sns.barplot(x=share_native_american,y=area_list,color="cyan",alpha=0.6,label="Native American")

sns.barplot(x=share_asian,y=area_list,color="yellow",alpha=0.6,label="Asian")

sns.barplot(x=share_hispanic,y=area_list,color="red",alpha=0.6,label="Hispanic")



ax.legend(loc="upper right",frameon=True)

ax.set(xlabel="Percentage of Races",ylabel="States",title="Percentage of State's Population over Races")

plt.grid()
# high school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_hs_ratio'] = sorted_data2['area_hs_ratio']/max( sorted_data2['area_hs_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_hs_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)



# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_hs_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
g=sns.jointplot(data.area_poverty_ratio,data.area_hs_ratio,kind="kde",size=7)    #kde:kernel density estimation

plt.savefig('graph.png')

plt.show()
g = sns.jointplot('area_poverty_ratio','area_hs_ratio',data=data,ratio=3,color='g',alpha=0.6)
kill.race.value_counts().index
kill.race.dropna(inplace=True)

labels=kill.race.value_counts().index

colors=['grey','blue','red','yellow','green','cyan']

explode=[0,0,0,0,0,0]

sizes=kill.race.value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title("Killed People According to Races",color="blue",fontsize=15)
sns.lmplot(x='area_poverty_ratio',y='area_hs_ratio',data=data)

plt.grid()
sns.kdeplot(data.area_poverty_ratio,data.area_hs_ratio,shade=True,cut=3,color='g')

plt.grid()
pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=data,palette=pal,inner='points')

plt.grid()
data.corr()
f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt=".1f", ax=ax)

plt.show()
plt.subplots(figsize =(10,10))

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.grid()
plt.subplots(figsize =(10,10))

sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)

plt.show()
kill.gender.value_counts()
sns.pairplot(data)

plt.show()
kill.manner_of_death.value_counts()
sns.countplot(kill.gender)

plt.title("gender",color="blue",fontsize=15)

plt.grid()
armed=kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel("Number of Weapons")

plt.xlabel("Weapon Types")

plt.title("Kill Weapon",color="blue",fontsize=15)

plt.grid()
above25=["above25"if i>= 25 else "below25" for i in kill.age]

df=pd.DataFrame({"age":above25})

sns.countplot(x=df.age)

plt.ylabel("Number of Killed People")

plt.title("Age of killed people",color="blue",fontsize=15)
sns.countplot(data=kill,x="race")

plt.title("race of killed people",color="blue",fontsize=15)
city=kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title("Most dangerous cities",color="blue",fontsize=15)