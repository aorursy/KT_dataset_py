# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
per_over = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding="Windows 1252")
per_people = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding="Windows 1252")
med = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding="Windows 1252")
share = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding = "Windows 1252")
police = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="Windows 1252")
per_people.head()
per_people.info()
per_people.poverty_rate.value_counts()
per_people["Geographic Area"].unique()
per_people.poverty_rate.replace("-",0.0,inplace=True)
per_people.poverty_rate = per_people.poverty_rate.astype(float)
area_list = list(per_people["Geographic Area"].unique())
area_pover_ratio = []

for i in area_list:
    x=per_people[per_people["Geographic Area"] == i]
    area_pover_rate=sum(x.poverty_rate) / len(x)
    area_pover_ratio.append(area_pover_rate)

    
data = pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_pover_ratio})
new_index = (data["area_poverty_ratio"].sort_values(ascending = False)).index.values
sorted_data = data.reindex(new_index)
#per_people.poverty_rate.value_counts()

# Visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data["area_list"], y = sorted_data["area_poverty_ratio"])
plt.xticks(rotation=  45)  # degree of States in gradus=45
plt.xlabel("States of the U.S.")
plt.ylabel("Poverty Ratio")
plt.title("Poverty Rate Given States")
plt.show()
per_people.info()
per_people.poverty_rate.value_counts()
police.head()
police.name.value_counts()
# Most Common 15 name or surname people
separate = police.name[police.name != "TK TK"].str.split()
#   FE: ("Tim","Elliot")
#          a      b
a,b=zip(*separate) #enter zip 
name_list = a+b
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
# x= names  y= amount of names                    x       y
x,y = zip(*most_common_names) #output zip: FE ("Michael",91)
x,y= list(x),list(y)

# Visualization through Graph in Bar Plot

plt.figure(figsize=(12,8))
sns.barplot(x,y, palette = sns.cubehelix_palette(len(x)))
plt.xlabel("Name or Surname of Killed People", color="red")
plt.ylabel("Frequency",color='blue')
plt.title("Most Common 15 Names or Surnames of Killed People with Guns",color="red")
plt.show()
per_over.head()
per_over.info()
#per_over.percent_completed_hs.value_counts()
# High school graduation rate of the population that is older than 25 in states
per_over.percent_completed_hs.replace("-",0.0, inplace = True)
per_over.percent_completed_hs = per_over.percent_completed_hs.astype(float)
area_list = list(per_over["Geographic Area"].unique())
area_hs = []

for i in area_list:
    x=per_over[per_over["Geographic Area"]==i]
    area_hs_rate = sum(x.percent_completed_hs)/len(x)
    area_hs.append(area_hs_rate)
# Sorting
data = pd.DataFrame({"area_list":area_list, "area_hs_ratio":area_hs})
new_index = (data['area_hs_ratio'].sort_values(ascending = True)).index.values
sorted_data2 = data.reindex(new_index)
# Visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2["area_list"],y = sorted_data2["area_hs_ratio"])
plt.xticks(rotation = 45)
plt.xlabel("States",color="blue")
plt.ylabel("High School Graduate Rate",color="blue")
plt.title("Percentage of Given State`s Population Above 25 That Has Graduated High School")
plt.show()
share.head()
share.info()
# Percentage of state`s population according to races that are black, white, native american, asian and hispanic
share.replace("-",0.0,  inplace=True)
share.replace("(X)",0.0,  inplace=True)

share.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]]=share.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)
area_list = list(share["Geographic area"].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for i in area_list:
    x=share[share["Geographic area"]==i]
    share_white.append(      sum(x.share_white)        / len(x))
    share_black.append(      sum(x.share_black)       / len(x))
    share_native_american.append( sum(x.share_native_american) / len(x))
    share_asian.append(      sum(x.share_asian)     / len(x))
    share_hispanic.append(   sum(x.share_hispanic) / len(x))
    
# Visualization
f,ax = plt.subplots(figsize=(9,15))
sns.barplot(x=share_white,           y=area_list, color="green",  alpha=0.5, label="White")
sns.barplot(x=share_black,           y=area_list, color="blue" ,  alpha=0.7, label="African")
sns.barplot(x=share_native_american, y=area_list, color="cyan",   alpha=0.7, label="Native American")
sns.barplot(x=share_asian,           y=area_list, color="yellow", alpha=0.9, label="Asian")
sns.barplot(x=share_hispanic,        y=area_list, color="red",    alpha=0.5, label="Hispanic")

ax.legend(loc="upper right", frameon = True)
ax.set(xlabel="Percengtage of Races", ylabel="States",title="Percentage of State`s Population")
plt.show()
# High school graduation rate vs Poverty rate of each state

# For Normanalize that number/max(number in data)
sorted_data["area_poverty_ratio"]=sorted_data["area_poverty_ratio"] / max(sorted_data["area_poverty_ratio"])
sorted_data2["area_hs_ratio"]= sorted_data2["area_hs_ratio"] / max(sorted_data2["area_hs_ratio"])

# Add to data
data= pd.concat([sorted_data,sorted_data2["area_hs_ratio"]],axis=1)
data.sort_values("area_poverty_ratio", inplace=True)

# Visualization
f,ax1= plt.subplots(figsize=(20,10))
sns.pointplot(x="area_list", y="area_poverty_ratio", data=data, color="lime", alpha=0.8)
sns.pointplot(x="area_list", y="area_hs_ratio",      data=data, color="red",  alpha=0.8)
plt.text(40,0.6, "high school graduate ratio",color="red",  fontsize=17,  style="italic")
plt.text(40,0.55,"poverty ratio",             color="lime", fontsize=18, style="italic")
plt.xlabel("States", fontsize=20, color="blue")
plt.ylabel("Values", fontsize=20, color="blue")
plt.title("High School Graduate VS Poverty Rate", fontsize=25, color="magenta")
plt.grid() # for background in line
plt.show()
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# Joint kernel density
# pearsonr = if it is 1, there is positive correlation and if it is, -1 there is negtive correlation
# If it is zero, there is no correlation between variable
# Show the joint distribution using kernel density estimation
sns.jointplot(data.area_poverty_ratio, data.area_hs_ratio, kind="kde", size=7)
plt.savefig("graph.png")
plt.show()
data.head()
# you can change parameters of joint plot
# kind: {"scatter" / "reg" / "resid" / "kde" / "hex"}
# Different usage of parameters but same plot with previous one
g=sns.jointplot("area_poverty_ratio", "area_hs_ratio", data=data, size=8, ratio=3, color="g")
police.race.head(15)
police.race.value_counts()
# W B H A N O is index
# 1201, 618, 423, 39, 31, 28 is unique
# Race rates according in kill data
police.race.dropna(inplace=True)
labels = police.race.value_counts().index
colors = ["grey","blue", "red", "yellow", "green", "brown"]
explode = [0,0,0,0,0,0]
sizes = police.race.value_counts().values

# Visualization
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%") 
# autopct == "3.4", "4.5"
plt.title("Killed People According to Races", color="blue", fontsize=15)
plt.show()
data.head()
# Visualization of school graduation rate vs Poverty rate of each state with different
# style of seaborn code
# Lmplot
# Show the results of a linear regression within each dataset

# Lmplot that we will use also Machine learning a lot. Again AGAIN

sns.lmplot(x="area_poverty_ratio", y="area_hs_ratio", data=data)
plt.show()
# Kernel Density Estimation
data.head()
# Visualization of high school graduation rate vs Poverty rate of each with different style
# of seaborn code
# Cubehelix plot
sns.kdeplot(data.area_poverty_ratio, data.area_hs_ratio, shade=True, cut=3)
plt.show()
data.head()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()
data.corr()
# Correlation map
# Visualization of high school graduation rate vs poverty rate of each state with different
# of seaborn code
f, ax= plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot=True, linewidths = 0.5, linecolor="gray", fmt=".1f", ax=ax)
plt.show()
police.head()
police.manner_of_death.unique()
# Manner of death (o`lim holati)
# Gender
# Age
# Plot the orbital period with horizontal boxes
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=police, palette="PRGn")
plt.show()
police.head()
# Swarm plot
# Manner of death
# Gender
# Age
sns.swarmplot(x="gender", y="age", hue="manner_of_death", data=police)
plt.show()
data.head()
# Pair plot
sns.pairplot(data)
plt.show()
police.gender.value_counts()
police.head()
# Kill properties
# Manner of death
sns.countplot(police.gender)
#sns.countplot(police.manner_of_death)
plt.title("Gender", color="blue", fontsize=15)
sns.countplot(police.manner_of_death)
plt.title("Manner of Death", color="red", fontsize=15)
armed = police.armed.value_counts()
print(armed)
armed[:7].index
# Kill weapon
armed = police.armed.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=armed[:7].index, y=armed[:7].values) # only 7 index that it takes
plt.xlabel("Weapon types")
plt.ylabel("Number of Weapon")
plt.title("Kill weapon", color="blue", fontsize=15)
plt.show()
# Age of Killed People
above25 = ["above25" if i>=25 else "below25" for i in police.age]
df = pd.DataFrame({"age":above25})
# Visualization
sns.countplot(x=df.age)
plt.xlabel("Age ", color = "blue", fontsize=15)
plt.ylabel("Number of Killed People", color="blue", fontsize=15)
plt.title("Age of killed people", color= "red", fontsize=20)
# Race of killed people
sns.countplot(data=police, x="race")
plt.title("Race of killed people", color="green", fontsize=15)
plt.show()
# Most Dangerous cities
city = police.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:12].index, y=city[:12].values)
plt.xticks(rotation=45)
plt.title("Most Dangerous Cities" ,color="brown", fontsize=15)

# Most Dangerous States
state=police.state.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=state[:20].index, y=state[:20].values)
plt.title("Most Dangerous State",color="blue", fontsize=15)

# Having mental illness or not for killed people
sns.countplot(police.signs_of_mental_illness)
plt.xlabel("Mental Illness")
plt.ylabel("Number of Mental Illness")
plt.title("Having mental illness or not", color="blue", fontsize=20)
plt.show()
# Threat types
sns.countplot(police.threat_level)
plt.xlabel("Threat types")
plt.title("Threat Types", color="green", fontsize=15)
# Flee types
sns.countplot(police.flee)
plt.xlabel("Flee Types")
plt.title("Flee Types", color="blue",fontsize=18)
# Having body cameras or not for police
sns.countplot(police.body_camera)
plt.xlabel("Having Body Cameras")
plt.title("Having body cameras or not on Police ", color="blue", fontsize=15)

# Kill numbers from states in kill data
sta = police.state.value_counts().index[:10]
sns.barplot(x=sta, y=police.state.value_counts().values[:10])
plt.title("Kill number from States", color="blue", fontsize=13)
