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



import warnings

warnings.filterwarnings('ignore') 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding='latin1')

percentage_people_below_poverty_level = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding='latin1')

percent_over_25_completed_highSchool = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding='latin1')

share_race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding='latin1')

kill = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding='latin1')
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.poverty_rate.value_counts()
# Bar Plot

percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())



area_poverty_ratio = []



for i in area_list:

    x = np.mean(percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i].poverty_rate)

    area_poverty_ratio.append(x)

    

data = pd.DataFrame({"area_list" : area_list,"area_poverty_ratio": area_poverty_ratio})

new_index = (data.area_poverty_ratio.sort_values(ascending = False)).index.values



sorted_data = data.reindex(new_index)



#Visualization

plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data.area_list,y=sorted_data.area_poverty_ratio)

plt.xticks(rotation = 45)

plt.ylabel("Poverty Rate")

plt.xlabel("States")

plt.title("Poverty Rate Given States")

plt.show()







kill.head()
kill.info()
kill.name
seperated = kill.name[kill.name != "TK TK"].str.split()

a, b = zip(*seperated)

name_list = a+b

counted = Counter(name_list)

most_common = counted.most_common(15)

x,y = zip(*most_common)

x,y = list(x),list(y)



#visualization

plt.figure(figsize = (15,10))

sns.barplot(x=x, y=y, palette = sns.cubehelix_palette(len(x)))

plt.xlabel("Name of Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most common 15 Name or Surname of killed people")

plt.show()







#seperated
#a,b
#name_list
#counted
#most_common
#x,y
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = percent_over_25_completed_highSchool["Geographic Area"].unique()

area_hs = []

for i in area_list:

    x = np.mean(percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"] == i].percent_completed_hs)

    area_hs.append(x)

#sorting

data = pd.DataFrame({'area_listem' : area_list, 'hsrate' : area_hs })

index = data.hsrate.sort_values(ascending = True).index.values #values puts indexes in an array

sorted_data2 = data.reindex(index)



#Visualize

plt.figure(figsize = (15,10))

sns.barplot(x=sorted_data2.area_listem, y=sorted_data2.hsrate)

plt.xticks(rotation = 45)

plt.ylabel("Frequency")

plt.xlabel("States")

plt.title("High school graduation rate of the population that is older than 25 in states")

plt.show()





    
share_race_city.head()
share_race_city.info()
share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace("(X)",0.0,inplace = True)

share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]] = share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

area_list = share_race_city["Geographic area"].unique()



share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

      

    share_white.append(np.mean(share_race_city[share_race_city["Geographic area"] == i].share_white))

    share_black.append(np.mean(share_race_city[share_race_city["Geographic area"] == i].share_black))

    share_native_american.append(np.mean(share_race_city[share_race_city["Geographic area"] == i].share_native_american))

    share_asian.append(np.mean(share_race_city[share_race_city["Geographic area"] == i].share_asian))

    share_hispanic.append(np.mean(share_race_city[share_race_city["Geographic area"] == i].share_hispanic ))

                          

#visualization

f, ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color = "green",alpha=0.5,label = "White")

sns.barplot(x=share_black,y=area_list,color = "blue",alpha=0.5,label = "Black")

sns.barplot(x=share_native_american,y=area_list,color = "cyan",alpha=0.5,label = "Native American")

sns.barplot(x=share_asian,y=area_list,color = "yellow",alpha=0.5,label = "Asian")

sns.barplot(x=share_hispanic,y=area_list,color = "red",alpha=0.5,label = "Hispanic")

plt.legend(loc = "lower right",frameon = True)

ax.set(xlabel ='Percentage of Races',ylabel = "States",title="Percentage of State's Population According to Races ")

plt.show()

            

                          

                          

                          
sorted_data.head()
sorted_data2.head()
sorted_data.area_poverty_ratio = sorted_data.area_poverty_ratio / max(sorted_data.area_poverty_ratio)

sorted_data2.hsrate = sorted_data2.hsrate  / max(sorted_data2.hsrate )

data3 = pd.concat([sorted_data,sorted_data2["hsrate"]],axis = 1)

data3.sort_values(by = "area_poverty_ratio", inplace = True)



#visualize



f,ax = plt.subplots(figsize = (20,10))



sns.pointplot(x = "area_list",y = "area_poverty_ratio", data = data3 ,color = "red",alpha = 0.8,label="Poverty Ratio")

sns.pointplot(x = "area_list",y = "hsrate", data = data3 ,color = "blue",alpha = 0.8,label = "High School Graduate Rate")

plt.text(39,0.6,"High School Graduate Ratio",color = "blue", fontsize = 17, style="italic")

plt.text(39,0.55,"Poverty Ratio",color = "red", fontsize = 17, style="italic")

plt.xlabel("States",color = "k",fontsize = 15)

plt.ylabel("Values",color = "k",fontsize = 15)

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

ax.legend()

plt.grid()
#f,ax = plt.subplots(figsize = (20,10))



#ax.plot_date(x = "area_list",y = "area_poverty_ratio", data = data3 ,color = "red",alpha = 0.8,label="Poverty Ratio",linestyle = "-")

#ax.plot_date(x = "area_list",y = "hsrate", data = data3 ,color = "blue",alpha = 0.8,label = "High School Graduate Rate",linestyle = "-")

#plt.xlabel("States",color = "k",fontsize = 15)

#plt.ylabel("Values",color = "k",fontsize = 15)

#plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

#ax.legend()

#plt.grid()

data3.head()
#Joint Plot

import scipy.stats as stats



j = sns.jointplot(x="area_poverty_ratio",y="hsrate",data=data3,kind="kde",size=7)

j.annotate(stats.pearsonr) #to show the pearsonr correlation stats

plt.show()
j = sns.jointplot(x="area_poverty_ratio",y="hsrate",data=data3,size=7,color = "r")

j.annotate(stats.pearsonr) #to show the pearsonr correlation stats

plt.show()
data3.head()
#with lm plot

sns.lmplot(x="area_poverty_ratio",y="hsrate",data=data3,size=7)

plt.show()
data3.head()
plt.subplots(figsize = (13,7))

sns.kdeplot(data3.area_poverty_ratio,data3.hsrate,shade= True, cut=5)

plt.show()
data3.head()
plt.subplots(figsize = (8,7))

paletim = sns.color_palette("Set1", n_colors=2, desat=0.7)

sns.violinplot(data=data3,inner="point",palette = paletim)

plt.show()
data3.head()
data3.corr()
f,ax = plt.subplots(figsize = (7,7))

ax = sns.heatmap(data3.corr(),annot = True,lw= 0.6, fmt='0.2f')
data3.head()
sns.pairplot(data3,size = 4)

plt.show()
kill.head()
kill.race.value_counts()
kill.race.dropna(inplace = True)

labels = ["White","Black","Hispanic","Asian","Native","Others"] #or easily kill.race.value_counts().index

colors =  ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','green','mediumpurple']

explode = (0,0.1,0,0,0,0)

sizes = kill.race.value_counts().values



#visualize



plt.figure(figsize = (10,10))

plt.pie(sizes, explode=explode,  labels=labels, colors=colors, autopct='%1.1f%%',shadow = True,startangle=160,wedgeprops={"edgecolor":"k",'linewidth': 1.5, 'antialiased': True},pctdistance = 0.7)

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

plt.legend()

plt.show()

kill.head()
f,ax = plt.subplots(figsize = (12,10))

ax = sns.boxplot(x="gender",y="age",hue="manner_of_death",data=kill,palette = "Set2")

plt.show()
kill.head()
f,ax = plt.subplots(figsize = (8,6))

ax = sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)

plt.show()