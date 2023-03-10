# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read datas

median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level['Geographic Area'].unique()
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
kill.name.value_counts()
# Most common 15 Name or Surname of killed people

separate = kill.name[kill.name != 'TK TK'].str.split() 

a,b = zip(*separate)                    

name_list = a+b                         

name_count = Counter(name_list)         

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)

# 

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
# High school graduation rate of the population that is older than 25 in states

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
share_race_city.head()
share_race_city.info()
#percentage of states population according to races that are black,white,native american, asian and hispanic 

share_race_city.replace(['-'],0.0,inplace=True)

share_race_city.replace(['(X)'],0.0,inplace=True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']]=share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_lis=list(share_race_city['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]

for i in area_list:

    x=share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))



#visualization

f,ax=plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list, color='green',alpha=0.5,label='white')

sns.barplot(x=share_black,y=area_list, color='blue',alpha=0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list, color='cyan',alpha=0.6,label='native american')

sns.barplot(x=share_asian,y=area_list, color='yellow',alpha=0.6,label='asian')

sns.barplot(x=share_hispanic,y=area_list, color='red',alpha=0.6,label='hispanic')



ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Percentage of races',ylabel='states',title='Percentage of States Popu??ation According to Races')

 
sorted_data.head()
sorted_data2.head()
#high school graduation rate vs poverty rate of each state



#normalization

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])



data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)





#visualization

f,ax=plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.6)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',aplha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize=17,style='italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize=18,style='italic')

plt.xlabel('States',fontsize=15,color='blue')

plt.ylabel('Values',fontsize=15,color='blue')

plt.title('High School Graduate VS Poverty Rate ',fontsize=30,color='blue')

plt.grid()







#visualizations of high school graduation rate vs poverty rate of each state with different style of seaborn code

#joint kernel density

#pearsonr=if it is 1, there is positive correlation,and if it is -1 there is negative correlation

#if it is zero,there is no correlation between variables

#show the joint distribution using kernel density estimation



g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind='kde',size=7)

plt.savefig('graph.png')

plt.show()

g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind='resid',size=7)

plt.savefig('graph.png')

plt.show()
#you can change parameters of jointplot

#kind:{"scatter"|"reg"|"resid"|"kde"|"hex"}

#different usage of parameters but same plot with pervious one



g=sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data,size=5,ratio=3,color='r')
kill.race.head(15)
kill.race.value_counts()
#race rates according in kill data

kill.race.dropna(inplace=True)

labels=kill.race.value_counts().index

colors=['grey','blue','red','yellow''green','brown']

explode=[0,0,0,0,0,0]

sizes=kill.race.value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")

plt.title('Killed people according to races ', color='blue',fontsize=15)

plt.show()
data.head()
#visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

#lmplot

#show the results of a linear redression within each dataset

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()
#visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

#cubehelix plot

sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=False,cut=3)

plt.show()
#show each distribution with both violins and points, use cubehelix to get a custom sequential paletted



pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=data,palette=pal,inner='points')

plt.show()
data.corr()
#corralation map (heatmap)

#visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code 

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidth=0.5,linecolor='red',fmt='.1f',ax=ax)

plt.show()

kill.head()
kill.manner_of_death.unique()
#manner of death(olum sekli): shot, shot and tasered

#gender

#age

#plot:the orbital period with horizontal boxes

sns.boxplot(x='gender',y='age',hue='manner_of_death',data=kill,palette='PRGn')

plt.show()
#swarm plot 

#manner of death:shot or shot and tasered

#gender 

#age

sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=kill,)

plt.show()
#pair plot

sns.pairplot(data)

plt.show()
#count plot



#kill properties

#manner of deaths

sns.countplot(kill.gender)

plt.title('gender',color='blue',fontsize=15)

plt.show()
#manner of deaths

sns.countplot(kill.manner_of_death)

plt.title('Manner of Deaths',color='blue',fontsize=15)

plt.show()

#kill weapon

armed=kill.armed.value_counts()

print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon types')

plt.title('Kill Weapon',color='blue',fontsize=15)

plt.show()

#age of killed people

above25=['above25'if i>=25 else 'below25' for i in kill.age]

df=pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color='blue',fontsize=15)

plt.show()
#race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color='blue',fontsize=15)

plt.show()
#most dangerous cities

city=kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:20].index,y=city[:20].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color='blue',fontsize=15)

plt.show()
#most dangerous states

state=kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index, y=state[:20].values)

plt.title('Most dangerous states', color='blue',fontsize=15)

plt.show()
#having metal illness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of mental illness')

plt.title('Having mental illness or not', color='blue',fontsize=15)

plt.show()







#threat types 

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.ylabel('Threat typess',color='blue',fontsize=15)

plt.show()
#flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee Types',color='blue',fontsize=15)

plt.show()
#having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Camera ')

plt.title('Having body camera or not on police',color='blue',fontsize=15)

plt.show()
#kill numbers from states in kill data

sta=kill.state.value_counts().index[:10]

sns.barplot(x=sta,y=kill.state.value_counts().values[:10])

plt.title('Kill numbers from states',color='blue',fontsize=15)

plt.show()