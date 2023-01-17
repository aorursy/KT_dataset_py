# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read datas

Percentage_People_Below_Poverty_Level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding="windows-1252")

Police_Killings_US = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding="windows-1252")

Percent_Over_25_Completed_HighSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding="windows-1252")

Share_Race_By_City = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding="windows-1252")
Percentage_People_Below_Poverty_Level.head()
Percentage_People_Below_Poverty_Level.info()
Percentage_People_Below_Poverty_Level.poverty_rate.value_counts()
Percentage_People_Below_Poverty_Level.poverty_rate.replace(["-"],0.0,inplace = True) # we fixed the meaningless value.
#Percentage_People_Below_Poverty_Level.poverty_rate.value_counts()
Percentage_People_Below_Poverty_Level.poverty_rate = Percentage_People_Below_Poverty_Level.poverty_rate.astype(float)

# we changed type of poverty_rate from object(str) to float.
#Percentage_People_Below_Poverty_Level.info()
Percentage_People_Below_Poverty_Level['Geographic Area'].unique()
area_list = list(Percentage_People_Below_Poverty_Level['Geographic Area'].unique()) #we get a list of unique states

area_poverty_ratio = list() # we created a list for ratio of each state
for i in area_list:  # we're in our unique state list

    area_filter = Percentage_People_Below_Poverty_Level[Percentage_People_Below_Poverty_Level['Geographic Area'] == i]

    area_poverty_rate = sum(area_filter.poverty_rate)/len(area_filter)

    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values #we sorted the poverty ratio values and get an index values array.

sorted_data = data.reindex(new_index) #with new_index array(sorted index values) we sorted the data

#visualization part (barplot)

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
Police_Killings_US.head()
Police_Killings_US.name.value_counts() # we can see, 'TK TK' is most probably a broken value so we must ignore it.
separated_names = Police_Killings_US.name[Police_Killings_US.name != 'TK TK'].str.split() 

#we separated the names and surnames

names,surnames= zip(*separated_names)

namelist = names+surnames

name_count = Counter(namelist)

most_common_names = name_count.most_common(15)

name,count = zip(*most_common_names)

name,count = list(name),list(count)
#visualization part

plt.figure(figsize=(15,10))

ax= sns.barplot(x=name, y=count,palette = sns.cubehelix_palette(len(names)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
Percent_Over_25_Completed_HighSchool.head()
Percent_Over_25_Completed_HighSchool.info()
Percent_Over_25_Completed_HighSchool.percent_completed_hs.value_counts()
Percent_Over_25_Completed_HighSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
#Percent_Over_25_Completed_HighSchool.percent_completed_hs.value_counts()
Percent_Over_25_Completed_HighSchool.percent_completed_hs = Percent_Over_25_Completed_HighSchool.percent_completed_hs.astype(float)
#Percent_Over_25_Completed_HighSchool.info()
area_list = list(Percent_Over_25_Completed_HighSchool['Geographic Area'].unique())

completed_hs_ratio = list()
for i in area_list:

    filter2 = Percent_Over_25_Completed_HighSchool[Percent_Over_25_Completed_HighSchool['Geographic Area'] == i]

    ratio_of_completed_hs = sum(filter2.percent_completed_hs)/len(filter2)

    completed_hs_ratio.append(ratio_of_completed_hs)
#sorting

data = pd.DataFrame({'area_list': area_list,'completed_hs_ratio':completed_hs_ratio})

new_index = (data['completed_hs_ratio'].sort_values(ascending=False)).index.values #we sorted the completed highschool ratio values and get an index values array.

sorted_data2 = data.reindex(new_index) #with new_index array(sorted index values) we sorted the data
#visualization part

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['completed_hs_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
Share_Race_By_City.head()
Share_Race_By_City.replace(['-'],0.0,inplace = True)

Share_Race_By_City.replace(['(X)'],0.0,inplace = True)
Share_Race_By_City.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = Share_Race_By_City.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(Share_Race_By_City['Geographic area'].unique())

share_black = list()

share_white = list()

share_asian = list()

share_native_american = list()

share_hispanic = list()
for i in area_list:

    area_filter = Share_Race_By_City[Share_Race_By_City['Geographic area']==i]

    share_white.append(sum(area_filter.share_white)/len(area_filter))

    share_black.append(sum(area_filter.share_black)/len(area_filter))

    share_asian.append(sum(area_filter.share_asian)/len(area_filter))

    share_native_american.append(sum(area_filter.share_native_american)/len(area_filter))

    share_hispanic.append(sum(area_filter.share_hispanic)/len(area_filter))
# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
# high school graduation rate vs Poverty rate of each state

#normalization

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

sorted_data2['completed_hs_ratio'] = sorted_data2['completed_hs_ratio']/max(sorted_data2['completed_hs_ratio'])
data = pd.concat([sorted_data,sorted_data2['completed_hs_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

data.head()
#visualization

f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color = 'lime',alpha=0.8)

sns.pointplot(x='area_list',y='completed_hs_ratio',data=data,color = 'red',alpha=0.8)

ax1.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

ax1.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data.area_poverty_ratio, data.completed_hs_ratio, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "completed_hs_ratio", data=data,size=5, ratio=3, color="r")
# Race rates according in kill data 

Police_Killings_US.race.dropna(inplace=True)

Police_Killings_US.race.value_counts()

labels = Police_Killings_US.race.value_counts().index

sizes = Police_Killings_US.race.value_counts().values

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

# visualization

plt.figure(figsize = (7,7))

plt.pie(sizes,explode = explode,labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio", y="completed_hs_ratio", data=data)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.area_poverty_ratio, data.completed_hs_ratio, shade=True, cut=3)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
data.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidths=.5,linecolor='red',fmt='.1f',ax=ax)

plt.show()
Police_Killings_US.head()
Police_Killings_US.manner_of_death.unique()
# Plot the orbital period with horizontal boxes

sns.boxplot(x="gender",y="age",hue="manner_of_death",data=Police_Killings_US,palette="PRGn")

plt.show()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=Police_Killings_US)

plt.show()
sns.pairplot(data)

plt.show()
Police_Killings_US.head()
Police_Killings_US.gender.value_counts()
sns.countplot(Police_Killings_US.gender)

plt.title("gender",color ='red',fontsize=15)

plt.show()
Police_Killings_US.armed.value_counts()
armed
# kill weapon

armed = Police_Killings_US.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
#age of killed people

above25 = ["above25" if i>25 else "below25" for i in Police_Killings_US.age]

sns.countplot(above25)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
#race of killed people

sns.countplot(Police_Killings_US.race)

plt.title('Race of killed people',color = 'blue',fontsize=15)
Police_Killings_US.head()
#most dangerous cities

city = Police_Killings_US.city.value_counts()

plt.figure(figsize=(11,6))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)
#most dangerous states

states = Police_Killings_US.state.value_counts()

plt.figure(figsize=(10,6))

sns.barplot(x=states[:18].index,y=states[:18].values)
#having mental illnes or not

sns.countplot(Police_Killings_US.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
#threat types

sns.countplot(Police_Killings_US.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)
#flee types

sns.countplot(Police_Killings_US.flee)

plt.xlabel('Flee types')

plt.title('Flee types',color='blue',fontsize=16)
#Having body camera or not

sns.countplot(Police_Killings_US.body_camera)

plt.xlabel('Having body camera')

plt.title('Having body camera',color = 'orange',fontsize=16)
#kill numbers

states2 = Police_Killings_US.state.value_counts().index[:10]

kills = Police_Killings_US.state.value_counts().values[:10]

sns.barplot(x=states2,y=kills)

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)