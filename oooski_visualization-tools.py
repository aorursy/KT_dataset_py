# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import scipy.stats as stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
median_house_hold_in_come = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

percentage_over_25_completed_highSchool = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")

share_race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding="windows-1252")

kill = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level.info()
area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())

area_poverty_ratio = []



for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i]

    area_poverty_rate = x.poverty_rate.mean()

    area_poverty_ratio.append(area_poverty_rate)

    

data = pd.DataFrame({"Area List": area_list,"area_poverty_ratio":area_poverty_ratio})

new_index = (data["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# Visualization



plt.figure(figsize=(20,20))

sns.barplot(x=sorted_data["Area List"],y = sorted_data["area_poverty_ratio"])

plt.xticks(rotation=90)  #degree of between x axis and x values

plt.xlabel("States")

plt.ylabel("Poverty Rate")

plt.title("Poverty Rate Given States")

plt.show()
separate = kill.name[kill.name != "TK TK"].str.split()

a,b = zip(*separate)

name_list = a+b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x = list(x)

y = list(y)



# Visualization



plt.figure(figsize =(15,15))

sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most common 15 Name or Surname of Killed People")

plt.show()
percentage_over_25_completed_highSchool.head()
percentage_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace=True)

percentage_over_25_completed_highSchool.percent_completed_hs = percentage_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list = percentage_over_25_completed_highSchool["Geographic Area"].unique()

area_highschool = []



for i in area_list:

    x = percentage_over_25_completed_highSchool[percentage_over_25_completed_highSchool["Geographic Area"] == i]

    area_highschool_rate = x.percent_completed_hs.mean()

    area_highschool.append(area_highschool_rate)

    

    

data = pd.DataFrame({"Area_List":area_list,"Area_HighSchool_Ratio":area_highschool})

data.sort_values(by = ["Area_HighSchool_Ratio"],ascending=True,inplace = True)



# Visualization



plt.figure(figsize=(15,20))

sns.barplot(x = data.Area_List, y = data.Area_HighSchool_Ratio)

plt.xticks(rotation = 90)

plt.xlabel("Name of States")

plt.ylabel("Area High School Ratio")

plt.title("Completed High School Ratio According to States")

plt.show()
share_race_city.info()
share_race_city.head()
share_race_city.replace(["(X)"],0.0,inplace=True)

share_race_city.replace(["-"],0.0,inplace=True)

share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]] = share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

area_list = list(share_race_city["Geographic area"].unique())



share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

    x = share_race_city[share_race_city["Geographic area"] == i]

    share_white.append(x.share_white.mean())

    share_black.append(x.share_black.mean())

    share_native_american.append(x.share_native_american.mean())

    share_asian.append(x.share_asian.mean())

    share_hispanic.append(x.share_hispanic.mean())
#Visualization

race_list = [share_white,share_black,share_native_american,share_asian,share_hispanic]

race_name_list = ["White","Black","Native","Asian","Hispanic"]

color_list = ["green","red","blue","cyan","black"]

alpha_list = list(np.arange(0.2,1.2,0.2))



plt.figure(figsize = (20,15))        

sayac = 0

           

for i in range(5):



    sns.barplot(x = race_list[sayac], y = area_list,color = color_list[sayac],alpha=alpha_list[sayac],label=race_name_list[sayac])

    sayac+=1



plt.xlabel("Percantage of Races",fontdict = {'fontsize':15,'color':'blue'})

plt.ylabel("States",fontdict = {'fontsize':15,'color':'blue'})

plt.title(label = "Percentages of State's Population According to Races",fontdict = {'fontsize':20,'color':'red'})

plt.legend(fontsize = 'xx-large',shadow=True)



plt.show()
# High School Graduation Rate vs Poverty Rate of Each State



sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"] / sorted_data['area_poverty_ratio'].max() # Normalization process

data.Area_HighSchool_Ratio = data.Area_HighSchool_Ratio / data.Area_HighSchool_Ratio.max() # Normalization process

data2 = pd.concat([sorted_data,data.Area_HighSchool_Ratio],axis=1)

data2.sort_values(by="area_poverty_ratio",inplace=True)

data2.rename(columns = {"Area List":"area_list"},inplace = True)

plt.figure(figsize = (20,15))



x = sns.pointplot(x = 'area_list',y = 'area_poverty_ratio' ,data = data2, color = 'lime',alpha = 0.8)

y = sns.pointplot(x = 'area_list',y = 'Area_HighSchool_Ratio',data = data2, color = 'red',alpha = 0.8)

plt.text(35,0.4,"High School Graduate Ratio",color="red",fontsize = 20)

plt.text(35,0.35,"Area Poverty Ratio",color="lime",fontsize = 20)

plt.xlabel("States",fontsize = 15,color="blue")

plt.ylabel("Values",fontsize = 15,color="blue")

plt.title("High School Graduate VS Poverty Rate",size=20,color="blue")

plt.grid()

plt.show()
# Visualizaton of high school graduation rate  vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr = if it is 1, there is a possitive correlation but it is -1 , there is a negative correlation.

# If it is zero, there is no correlation between variables.

# Show the joint distribution using kernel densty estimation.(kde)

g = sns.jointplot(data2.area_poverty_ratio,data2.Area_HighSchool_Ratio,kind="kde",height=10)

g.annotate(stats.pearsonr)

plt.show()
g = sns.jointplot(data2.area_poverty_ratio,data2.Area_HighSchool_Ratio,kind="scatter",height=10,ratio=5,color='r')
# Race rates according in kill data

kill.race.dropna(inplace=True)

labels = kill.race.value_counts().index

colors = ["grey","blue","red","yellow","green","brown"]

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



# visualization



plt.figure(figsize = (20,15))

plt.pie(sizes,explode = explode,labels = labels,colors = colors,autopct = "%1.1f%%")

plt.title("Killed People According to Races",color="blue",size=20)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code.

# lmplot

# Show the results of lenear regression within each dataset

sns.lmplot(x = "area_poverty_ratio",y = "Area_HighSchool_Ratio",data = data2,size=15)
# Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

# Cubehelix plot 

plt.figure(figsize=(20,15))

sns.kdeplot(data2.area_poverty_ratio,data2.Area_HighSchool_Ratio,shade=True,cut = 5)

plt.xlabel("Area Poverty Ratio",size=15)

plt.ylabel("Area High School Ratio",size=15)

plt.title("Correlation between poverty and high school graduation ratios",size=20)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom squential palette

plt.figure(figsize = (15,15))

pal = sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data = data2, palette = pal, inner = "point")

plt.show()
plt.figure(figsize = (10,10))

sns.heatmap(data2.corr(),annot=True,linewidth=.5,fmt='.1f')

plt.show()
# Manner of death (the way to death) : been shot, both been shot and tasered

# Gender

# Age

# Plot the orbital period with horizontal boxes

plt.figure(figsize = (10,10))

sns.boxplot(x='gender',y='age', hue = "manner_of_death", data = kill, palette = "GnBu")

plt.show()
# Swarm plot

# Manner of death

# gender

# age

plt.figure(figsize = (15,15))

sns.swarmplot(x='gender',y="age",hue="manner_of_death",data = kill)

plt.xlabel("Gender",size=20)

plt.ylabel("Age",size=20)

plt.show()
sns.pairplot(data2,height=5)

plt.show()
# Kill properties

# Manner of Death

plt.figure(figsize=(10,10))

sns.countplot(kill.gender)

sns.countplot(kill.manner_of_death)

plt.title("Manner of Death",color='blue',size=15)
plt.figure(figsize=(10,10))

sns.countplot(kill.gender)

plt.title("Gender",color='blue',size=15)