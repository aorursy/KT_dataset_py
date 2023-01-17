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

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings('ignore') 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come=pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding="windows-1252")

percentage_people_below_poverty_level=pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

percent_over_25_completed_highSchool=pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")

share_race_city=pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding="windows-1252")

kill=pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level["Geographic Area"].unique()
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
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace=True)

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list=percentage_people_below_poverty_level["Geographic Area"].unique()

area_poverty_ratio = []

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})

new_index=(data["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)



#Visualization:

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["area_list"],y=sorted_data["area_poverty_ratio"])

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace=True)

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list=percentage_people_below_poverty_level["Geographic Area"].unique()

area_poverty_ratio = []

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})

new_index=(data["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)

sorted_data.head()
kill.head()
kill.name.value_counts()
seperate=kill.name[kill.name != "TK TK"].str.split()

a,b=zip(*seperate)

name_list=a+b

name_count=Counter(name_list)

most_common_names=name_count.most_common(15)

x,y=zip(*most_common_names)

x,y = list(x),list(y)



#Visualization

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
#split():

"mehmet apalan".split()
#a+b
#Counter(name_list)  
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
# High school graduation rate of the population that is older than 25 in states

percent_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list=list(percent_over_25_completed_highSchool["Geographic Area"].unique())

area_highschool=[]

for i in area_list:

    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    area_highschool_rate=sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

#Sorting

data=pd.DataFrame({"area_list":area_list,"area_highschool_ratio":area_highschool})

new_index=(data["area_highschool_ratio"].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)

#Visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2["area_list"],y=sorted_data2["area_highschool_ratio"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("High School Graduate Rate")

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School",color="r")

plt.grid(True)

plt.show()
percent_over_25_completed_highSchool.head(5)
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

    x=share_race_city[share_race_city["Geographic area"]==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

#visualization

f,ax=plt.subplots(figsize=(9,15))

sns.barplot(x=share_white,y=area_list,color="y",alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='k',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='b',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='g',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True) #frameon=True diyerek sağ alttaki legendın görünmesini sağladık, false dersek arkası saydam olur,sadece yazılar olur.

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
sorted_data.head()
sorted_data2.head()
x=[1,2,3,4,5]

y1=[1,2,3,4,5]

y2=[1000,900,800,700,600]

plt.plot(x,y1)

plt.plot(x,y2)

plt.show()
plt.plot(x,y1)
# high school graduation rate vs Poverty rate of each state

sorted_data["area_poverty_ratio"]=sorted_data["area_poverty_ratio"]/max( sorted_data['area_poverty_ratio']) #normalization

sorted_data2["area_highschool_ratio"]=sorted_data2["area_highschool_ratio"]/max( sorted_data2['area_highschool_ratio']) #normalization

data2=pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1) #columnları yan yana concat ediyoruz.

data2.sort_values('area_poverty_ratio',inplace=True) #4area_poverty_ratio'ya göre sıralıyoruz.



#visualize

f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x="area_list",y="area_poverty_ratio",data=data2,color="lime",alpha=.8) #datamızı data eşittir diyerek...

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data2,color='red',alpha=0.8) #...x y y oradan direkt çekiyoruz.

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize=17,style="italic") # metnin konumunu seçiyoruz.

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
data2.head()
data2.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g=sns.jointplot(data2.area_poverty_ratio, data2.area_highschool_ratio,kind="kde",size=8)

plt.savefig('graph.png')

plt.show()
data2.head()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

g=sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data2,size=5, ratio=5, color="r")
g=sns.jointplot(data2.area_poverty_ratio, data2.area_highschool_ratio,kind="reg",size=4)

plt.show()
g=sns.jointplot(data2.area_poverty_ratio, data2.area_highschool_ratio,kind="scatter",size=4)

plt.show()
g=sns.jointplot(data2.area_poverty_ratio, data2.area_highschool_ratio,kind="resid",size=4)

plt.show()
g=sns.jointplot(data2.area_poverty_ratio, data2.area_highschool_ratio,kind="hex",size=4)

plt.show()
kill.race.head(10)
kill.race.value_counts()
# Race rates according in kill data 

kill.race.dropna(inplace=True) #içi boş olanları çıkardık

labels=kill.race.value_counts().index # w,b,h vs.yi index olarak aldık.

colors=["grey","b","r","y","g","brown"]

explode=[0,0,0,0,0,0] #burada piechartlaarın oranı olacak, sonradan doldurucaz.

sizes=kill.race.value_counts().values

# visual

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=labels,explode=explode,colors=colors,autopct="%1.1f%%") #autopct virgülden sonra kaç basamak...

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

plt.show()
data2.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio",data=data2)

plt.grid()

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data2.area_poverty_ratio,data2.area_highschool_ratio,shade=True,cut=5)

plt.show()
#shade=False:

sns.kdeplot(data2.area_poverty_ratio,data2.area_highschool_ratio,shade=False,cut=5)

plt.show()
#cut???

sns.kdeplot(data2.area_poverty_ratio,data2.area_highschool_ratio,shade=True,cut=1)

plt.show()
data2.head()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=data2,palette=pal,inner="points")

plt.show()
data2.corr()
#correlation map

#Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data2.corr(),annot=True,linewidths=.5,linecolor="r",fmt=".1f",ax=ax)

plt.show()
kill.head()
kill.manner_of_death.unique()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

# Plot the orbital period with horizontal boxes

f,ax=plt.subplots(figsize=(12,8))

sns.boxplot(x="gender",y="age",hue="manner_of_death",data=kill,palette="PRGn")

plt.show()
kill.head()
# swarm plot

# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

f,ax=plt.subplots(figsize=(11,11))

sns.swarmplot(x="gender",y="age",data=kill,hue="manner_of_death")

plt.show()
data.head()
sns.pairplot(data)
kill.gender.value_counts()
kill.head()
# kill properties

# Manner of death

sns.countplot(kill.gender)

plt.title("gender",color = 'blue',fontsize=15)
sns.countplot(kill.manner_of_death)
age25=["above25" if i>25 else "below25" if i<25 else "equal25"for i in kill.age]

df=pd.DataFrame({"age":age25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color="b",fontsize=15)
sns.countplot(data=kill,x="race")

plt.title('Race of killed people',color="b",fontsize=15)
sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)
sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)
sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)
#1

armed=kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
#2

city=kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title("Most dangerous cities",color="r",fontsize=15)
#3

state=kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'r',fontsize=15)
#4

state2 = kill.state.value_counts().index[:10]

plt.figure(figsize=(10,7))

sns.barplot(x=state2,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)