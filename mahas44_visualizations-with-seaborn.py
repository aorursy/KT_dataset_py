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

print(check_output(["ls","../input"]).decode("utf8"))



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read Datas



median_house_hold_in_come = pd.read_csv("../input/MedianHouseholdIncome2015.csv", encoding = "windows-1252")

percentage_people_below_poverty_level = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv", encoding = "windows-1252")

percent_over_25_completed_high_school = pd.read_csv("../input/PercentOver25CompletedHighSchool.csv", encoding = "windows-1252")

police_killings_US = pd.read_csv("../input/PoliceKillingsUS.csv", encoding = "windows-1252")

share_race_by_city = pd.read_csv("../input/ShareRaceByCity.csv", encoding = "windows-1252")



percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level["Geographic Area"].unique()
# Poverty rate of each state

# percentage_people_below_poverty_level.poverty_rate.value_counts()



# we are assign 0 to "-". Because "-" dosen't make sense 

percentage_people_below_poverty_level.poverty_rate.replace(["-"], 0.0, inplace=True) 

# poverty rate assigned object and we are converting object to float

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)



area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())

area_poverty_ratio = []



# sort data high to low value 

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)



data = pd.DataFrame({"area_list": area_list,"area_poverty_ratio":area_poverty_ratio})

new_index = (data["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization (Bar Plot)

plt.figure(figsize= (20,10))

sns.barplot(x = sorted_data.area_list, y = sorted_data.area_poverty_ratio)

plt.xticks(rotation = 90)

plt.xlabel("States")

plt.ylabel("Poverty Rate")

plt.title("Poverty Rate Given States")
police_killings_US.head()
# police_killings_US.name.value_counts ()
# Most common 15 name or surname of killed people



seperate = police_killings_US.name[police_killings_US.name != "TK TK"].str.split()

a,b = zip(*seperate)

name_list = a+b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)



# visulization



plt.figure(figsize=(20,10))

ax = sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most common 15 Name or Surname of killed people")

percent_over_25_completed_high_school.info()
percent_over_25_completed_high_school.head()
percent_over_25_completed_high_school.percent_completed_hs.value_counts()
# High school graduation rate of thepopulation that is older than 25 years old in states



percent_over_25_completed_high_school.percent_completed_hs.replace(["-"],0.0,inplace=True)

percent_over_25_completed_high_school.percent_completed_hs =percent_over_25_completed_high_school.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_high_school["Geographic Area"].unique())

area_high_school = []



for i in area_list:

    x = percent_over_25_completed_high_school[percent_over_25_completed_high_school["Geographic Area"] == i]

    area_high_school_rate = sum(x.percent_completed_hs)/len(x)

    area_high_school.append(area_high_school_rate)



 # sorting

data = pd.DataFrame({"area_list":area_list,"area_high_school_ratio":area_high_school})

new_index = (data.area_high_school_ratio.sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# visualization

plt.figure(figsize=(20,10))

ax = sns.barplot(x = sorted_data2.area_list,y=sorted_data2.area_high_school_ratio)

plt.xticks(rotation = 90)

plt.xlabel("States")

plt.ylabel("High School Graduate Rate")

plt.title("Percentage of Given State's Population Above 25 That Has Graduate High School ")

    
share_race_by_city.head()
share_race_by_city.info()
# Percentage of state's population according to races that are black,white,asian,hispanic,native american

share_race_by_city.replace(["(X)"],0.0,inplace=True)



share_race_by_city.share_white = share_race_by_city.share_white.astype(float)

share_race_by_city.share_black = share_race_by_city.share_black.astype(float)

share_race_by_city.share_native_american = share_race_by_city.share_native_american.astype(float)

share_race_by_city.share_asian = share_race_by_city.share_asian.astype(float)

share_race_by_city.share_hispanic = share_race_by_city.share_hispanic.astype(float)



area_list = list(share_race_by_city["Geographic area"].unique())

share_white_list = []

share_black_list = []

share_asian_list = []

share_hispanic_list = []

share_native_american_list = []



for i in area_list:

    x = share_race_by_city[share_race_by_city["Geographic area"] == i]

    share_white_list.append(sum(x.share_white)/len(x))

    share_black_list.append(sum(x.share_black)/len(x))

    share_asian_list.append(sum(x.share_asian)/len(x))

    share_hispanic_list.append(sum(x.share_hispanic)/len(x))

    share_native_american_list.append(sum(x.share_native_american)/len(x))

    

# visualization



f,ax = plt.subplots(figsize=(10,15))

sns.barplot(x = share_white_list, y = area_list,color = "green", alpha = 0.4, label= "White")

sns.barplot(x = share_black_list, y = area_list, color = "yellow",alpha = 0.5, label="African American")

sns.barplot(x = share_asian_list, y = area_list, color = "red",alpha = 0.6, label="Asian")

sns.barplot(x = share_hispanic_list, y = area_list, color = "blue",alpha = 0.4, label="Hispanic")

sns.barplot(x = share_native_american_list, y = area_list, color = "magenta",alpha = 0.6, label="Native American")



ax.legend(loc="lower right", frameon = True) # Legend

ax.set(xlabel="Percentage of Races",ylabel="States",title="Percentage of State's Population According to Races")
# High school graduation rate vs poverty rate of each state



sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"]) #normalization

sorted_data2["area_high_school_ratio"] = sorted_data2["area_high_school_ratio"]/max(sorted_data2["area_high_school_ratio"]) #normalization

data = pd.concat([sorted_data,sorted_data2["area_high_school_ratio"]],axis=1)

data.sort_values("area_poverty_ratio",inplace=True)



#visualization

f, ax1 = plt.subplots(figsize=(20,10)) #x-axis length=20, y-axis length=10

sns.pointplot(x="area_list", y="area_poverty_ratio", data=data, color="purple", alpha=0.8 )

sns.pointplot(x="area_list", y="area_high_school_ratio", data=data, color="red", alpha=0.8)

plt.text(40, 0.66, "high school graduate ratio", color="purple",fontsize=18, style="italic")

plt.text(40, 0.55, "poverty ratio", color="red", fontsize=18, style="italic")

plt.xlabel("States",fontsize=15, color="blue")

plt.ylabel("Values",fontsize=15, color="blue")

plt.title("High School Graduate   VS   Poverty Ratio", fontsize=22, color="green")

plt.grid()

# Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr = if it is 1, there is positive correlation and if it is -1, there is negative correlation

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation

from scipy import stats

g= sns.jointplot(data.area_poverty_ratio, data.area_high_school_ratio, kind="kde", height=7)

g = g.annotate(stats.pearsonr)

plt.savefig("graph.png")

plt.show()
g = sns.jointplot("area_poverty_ratio","area_high_school_ratio", data=data, height=5, ratio=3, color="red")

g = g.annotate(stats.pearsonr)
police_killings_US.race.value_counts()
# Race rates according in kill data

police_killings_US.race.dropna(inplace=True)

labels = police_killings_US.race.value_counts().index

colors = ["grey","blue","red","yellow","green","brown"]

explode = [0,0,0,0,0,0]

sizes = police_killings_US.race.value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(sizes, explode=explode, labels=labels, colors = colors, autopct="%1.1f%%")

plt.title("Killed People According to Races", color= "Blue", fontsize=15)
# Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

# lmplot

# Show the results of a linear regression within each dataset



sns.lmplot(x="area_poverty_ratio", y="area_high_school_ratio", data=data)

plt.show()
# Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

# cubehelix plot



sns.kdeplot(data.area_poverty_ratio,data.area_high_school_ratio, shade=True, cut=2)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette



pal = sns.cubehelix_palette(2, rot=-0.5, dark=0.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
# correlation map

# Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code



f,ax= plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(), annot=True,linecolor="green", linewidths=0.5, fmt="0.1f", ax=ax)

plt.show()
#police_killings_US.manner_of_death.unique()

police_killings_US.head()
# manner of death : shot or shot and Tasered

# Plot the orbital period with horizontal boxes



sns.boxplot(x="gender",y="age", hue="manner_of_death", data=police_killings_US, palette="PRGn")

plt.show()
# swarm plot

# manner of death : shot or shot and Tasered



sns.swarmplot(x="gender", y="age", hue="manner_of_death", data=police_killings_US)

plt.show()
data.head()
# pair plot



sns.pairplot(data)

plt.show()
# kill properties

# Manner of death



sns.countplot(police_killings_US.gender)

plt.title("Gender", color ="blue", fontsize=14)

sns.countplot(police_killings_US.manner_of_death)

plt.title("Manner of Death", color ="blue", fontsize=14)

# kill weapon



armed = police_killings_US.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index, y= armed[:7].values)

plt.ylabel("Number of Weapon")

plt.xlabel("Weapon Types")

plt.title("Kill Weapon", color="blue",fontsize = 14)
# age of killed people



above25 = ["above25" if i > 25 else "below25" for i in police_killings_US.age]

df = pd.DataFrame({"age":above25})

sns.countplot(x=df.age)

plt.ylabel("Number of killed People")

plt.title("Age of Killed People", color="blue", fontsize=14)