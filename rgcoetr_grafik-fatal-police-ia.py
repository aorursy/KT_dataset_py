# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

#patplotlib inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #butun datayi utf8 karakterine uygun mu degil mi kontrol eder.



# Any results you write to the current directory are saved as output.
# Read datas

median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")



percentage_people_below_poverty_level.head()

percentage_people_below_poverty_level.info()

percentage_people_below_poverty_level.describe()
percentage_people_below_poverty_level.poverty_rate.value_counts()

#sayilarin bilgilerini verdi
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.poverty_rate.replace(["-"], 0.0, inplace = True)

#tum datadaki "-" leri "0.0" yaptik

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

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 77)

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

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

# "-" leri 0.0 yaptik

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

# percent_completed_hs sutunun floata cevirdik. daha once object i yani str
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

plt.xticks(rotation= 90) # alta yazilacak isimler 90 derece aciyla gosterir

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_city.head()

share_race_city.info()

# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

#butun sutunlardaki degerleri float yaptik
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

f,ax = plt.subplots(figsize = (9,15)) #cubuklarin ince kalinliklarini ve uzunluklarini ayarlar

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' ) #alpha saydamligi belirler

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
# high school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)
# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic') # ustundeki bilgilendirmenin konumunu belirler

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic') # ustundeki bilgilendirmenin konumunu belirler

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
data.head()

###Join Plot cizimi

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=11)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
########Pie Chart



kill.race.head(15)

kill.race.value_counts()

# Race rates according in kill data 

kill.race.dropna(inplace = True)

#none lari atacak

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values

# Pasta Cizimi

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%') 

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

##### Lm Plot

data.head()

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio",  data=data)

plt.show()
### KDE cizimi

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=2)

plt.show()
###########Violin Plot¶

data.head()

# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
# Heatmap correlation cizimi



data.corr()



#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax) #ftm noktadan sonra kac hane olacak onu verir

plt.show()

#### Box Plot

kill.head()

kill.manner_of_death.unique()

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()
# Swarm Plot



kill.head()

# swarm plot

# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()
# Pair Plot¶

#sayisal degerleri alir ve cizer otomatik secer

sns.pairplot(data)

plt.show()
# Count Plot¶

kill.gender.value_counts()
# kill properties

# Manner of death

sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)
# kill weapon

armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:5].index,y=armed[:5].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people

above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)
# Most dangerous cities

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=50)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)
# most dangerous states

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)

# Threat types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)
# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)