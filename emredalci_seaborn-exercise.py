# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read datas

percent_over_25_completed_highschool = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv" , encoding ="windows-1252")

median_household_income_2015 = pd.read_csv("../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv" , encoding ="windows-1252" )

police_killings_us = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding ="windows-1252")

share_race_by_city = pd.read_csv("../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding ="windows-1252")

percentage_people_below_poverty_level = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding ="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level["Geographic Area"].unique()
len(percentage_people_below_poverty_level["Geographic Area"].unique())
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0, inplace = True) # - values are changed as 0.0

#percentage_people_below_poverty_level.poverty_rate.replace(["-"],None, inplace = True) # or we can change - values to Nan values

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float) # The object values that are in poverty level are changed as float

area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())

area_poverty_ratio = []

for i in area_list:

    """

    We determine AL state(1). Then we calculate average poverty rate(2). And then we append area_poverty_rate to area_poverty_ratio (3).

    If you write print(i) and print(x) separately, you can see what values you have

    """

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i] # 1 , we create new dataframe for each state

    area_poverty_rate = sum(x.poverty_rate)/len(x) # 2

    area_poverty_ratio.append(area_poverty_rate) # 3

data = pd.DataFrame({"area_list": area_list,"area_poverty_ratio": area_poverty_ratio }) # we create new data that is from area_list and area_poverty_ratio

new_index = (data["area_poverty_ratio"].sort_values(ascending=False)).index.values # we sort area_poverty_ratio as index values, ascending determine that sorted values increase or decrease

sorted_data = data.reindex(new_index) # we change index values of data. And we put new data to sorted_data



#visualization

plt.figure(figsize = (15,10)) # size of figure

sns.barplot(x = sorted_data["area_list"], y = sorted_data["area_poverty_ratio"]) # we create barplot

plt.xticks(rotation= 45) # angle of x values

plt.xlabel("States")

plt.ylabel("Poverty Rate")

plt.title("Poverty Rate Given States")
police_killings_us.head()
police_killings_us.info()
police_killings_us.name.value_counts()
# Most common 15 Name or Surname of killed people

separate = police_killings_us.name[police_killings_us.name != "TK TK"].str.split() #  split name and surname except for "TK TK"

a,b = zip(*separate) #  unzip separate data

name_list = a+b #  type(name_list) = tupple

name_count = Counter(name_list) # count names . type(name_count) = collections.Counter

most_common_names = name_count.most_common(15) # determine most 15 common names that is list

x,y = zip(*most_common_names) # unzip list as x and y. (x and y are tupple)

x,y = list(x),list(y) # change tupple to list

#visualization

plt.figure(figsize=(15,10))

sns.barplot(x=x,y=y, palette = sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most common 15 Name or Surname of killed people")
percent_over_25_completed_highschool.head()
percent_over_25_completed_highschool.info()
percent_over_25_completed_highschool.percent_completed_hs.value_counts()
# High school graduation rate of the population that is older than 25 in states

percent_over_25_completed_highschool.percent_completed_hs.replace(["-"],None,inplace = True)

percent_over_25_completed_highschool.percent_completed_hs = percent_over_25_completed_highschool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highschool["Geographic Area"].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highschool[percent_over_25_completed_highschool["Geographic Area"]== i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

data2 = pd.DataFrame({"area_list": area_list, "area_highschool_ratio": area_highschool})

new_index2 = (data2["area_highschool_ratio"].sort_values(ascending = True)).index.values

sorted_data2 =data2.reindex(new_index2)

#visualization

plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data2["area_list"], y =sorted_data2["area_highschool_ratio"] )

plt.xticks(rotation = 90)

plt.xlabel("States")

plt.ylabel("High School Graduate Rate")

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_by_city.head()
share_race_by_city.info()
share_race_by_city.share_white.value_counts() 

#share_race_by_city.share_black.value_counts()

#share_race_by_city.share_native_american.value_counts()

#share_race_by_city.share_asian.value_counts()

#share_race_by_city.share_hispanic.value_counts()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_by_city.replace(["-"],None, inplace = True)

share_race_by_city.replace(["(X)"],None, inplace = True)

share_race_by_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]] =share_race_by_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

area_list = list(share_race_by_city["Geographic area"].unique())

share_white =[]

share_black =[]

share_native_american = []

share_asian =[]

share_hispanic =[]

for i in area_list:

    x = share_race_by_city[share_race_by_city["Geographic area"] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

#visualization

f, ax =plt.subplots(figsize = (9,15))

sns.barplot(x =share_white, y = area_list, color = "green", alpha = 0.5,  label = "white" )

sns.barplot(x =share_black, y = area_list, alpha = 0.5, color = "black", label = "black" )

sns.barplot(x =share_native_american, y = area_list, alpha = 0.5, color = "cyan", label = "native" )

sns.barplot(x =share_asian, y = area_list, alpha = 0.5, color = "yellow", label = "asian" )

sns.barplot(x =share_hispanic, y = area_list, alpha = 0.5, color = "red", label = "hispanic" )

ax.legend(loc='lower right',frameon = True) 

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
# high school graduation rate vs Poverty rate of each state

sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/ max(sorted_data["area_poverty_ratio"]) #Basic Normalization

sorted_data2["area_highschool_ratio"] = sorted_data2["area_highschool_ratio"]/ max(sorted_data2["area_highschool_ratio"]) #Basic Normalization

data3 = pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]], axis = 1)

data3.sort_values("area_poverty_ratio", inplace = True)

#visualization

f ,ax = plt.subplots(figsize = (20,10))

sns.pointplot ( x = "area_list", y = "area_poverty_ratio", data =data3, color = "green") # if we add data parameter in sns.pointplot , python recognize data columns

sns.pointplot ( x = "area_list", y = "area_highschool_ratio" ,data=data3, color = "red")

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio",data=data3, kind="kde", size = 7)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data3,size=5, ratio=3, color="r")
police_killings_us.race.head(15)
police_killings_us.info()
police_killings_us.race.value_counts()
# Race rates according in kill data 

labels = police_killings_us.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0.1,0,0,0,0] #offsetting a slice with "explode"

sizes = police_killings_us.race.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
data3.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data3)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data3.area_poverty_ratio, data3.area_highschool_ratio, shade=True, cut=3) # cut size of shape
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data3, palette=pal, inner="points")

plt.show()
data3.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data3.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
# manner of death

# gender

# age

# Plot the orbital period with horizontal boxes

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=police_killings_us, palette="PRGn")
# swarm plot

# manner of death

# gender

# age

sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=police_killings_us)
# pair plot

sns.pairplot(data3)
police_killings_us.manner_of_death.value_counts()
# kill properties

# Manner of death

sns.countplot(police_killings_us.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)
# kill weapon

armed = police_killings_us.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people

filter25 = ["above_25" if i >= 25 else "below25" for i in police_killings_us.age]

data4 = pd.DataFrame({"age": filter25})

sns.countplot(x = data4.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
# Most dangerous cities

city = police_killings_us.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)
# most dangerous states

state = police_killings_us.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)
# Having body cameras or not for police

sns.countplot(police_killings_us.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)