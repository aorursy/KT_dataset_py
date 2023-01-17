# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
per_poverty = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding = "windows-1252")

race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding = "windows-1252")

median_income = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding = "windows-1252")

per_25_highschool = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding = "windows-1252")

police_killings_us = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding = "windows-1252")

# we need to encode as "encoding = "windows-1252" for this error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x88 in position 8: invalid start byte
per_poverty.info()

# per_poverty dataframe has a 3 features;  Geographic Area, City and Poverty Rate.
# showing first 5 rows

per_poverty.head()
# The "unique()" function is used to find the unique elements of an array.

per_poverty["Geographic Area"].unique()

# Geographic Area's keys abbreviated name of US's states for example;

# AL : Alabama, AK: Alaska, AZ: Arizona, AR: Arkansas, CA: California...
per_poverty.poverty_rate = per_poverty.poverty_rate.replace("-","0.0")

per_poverty.poverty_rate = per_poverty.poverty_rate.astype(float)

area_list = list(per_poverty["Geographic Area"].unique())

area_poverty_ratio = []

for i in area_list:

    x = per_poverty[per_poverty["Geographic Area"]==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

df = pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})

new_index = (df["area_poverty_ratio"].sort_values(ascending=False)).index.values

sorted_data = df.reindex(new_index)

plt.figure(figsize= (15,10))

sns.barplot(x= sorted_data["area_list"], y = sorted_data["area_poverty_ratio"],palette = sns.cubehelix_palette(len(x),reverse = True))

plt.show()
per_25_highschool.head()
per_25_highschool.percent_completed_hs = per_25_highschool.percent_completed_hs.replace("-","0.0")

per_25_highschool.percent_completed_hs = per_25_highschool.percent_completed_hs.astype(float)

area_list_2 = list(per_25_highschool["Geographic Area"].unique())

completed_level = []

for i in area_list_2:

    x = per_25_highschool[per_25_highschool["Geographic Area"]==i]

    complete_ratio = sum(x.percent_completed_hs)/len(x)

    completed_level.append(complete_ratio)

df_2 = pd.DataFrame({"Area List":area_list_2,"Complete Ratio":completed_level})

new_index = (df_2["Complete Ratio"].sort_values(ascending=True)).index.values

sorted_df = df_2.reindex(new_index)



plt.figure(figsize= (15,10))

sns.barplot(x = sorted_df["Area List"], y = sorted_df["Complete Ratio"])

plt.title("Percent Over 25 Completed HighSchool")

plt.xlabel("States")

plt.show()
police_killings_us.head()

from collections import Counter



separate = police_killings_us.name[police_killings_us.name != "TK TK"].str.split()

a,b = zip(*separate)

name_list = a

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x), list(y)

plt.figure(figsize= (15,10))

sns.barplot(x= x, y = y, palette= "rocket")

plt.title("Most names used in people killed")

plt.xlabel("Names")

plt.ylabel("Frequency")

plt.show()
surname_list = b

surname_count = Counter(surname_list)

most_common_surname = surname_count.most_common(15)

x,y = zip(*most_common_surname)

x,y = list(x),list(y)

plt.figure(figsize= (15,10))

sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(len(x), reverse = True))

plt.title("Most surnames used in people killed")

plt.xlabel("Surnames")

plt.ylabel("Frequency")

plt.show()
race_city.info()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

race_city.replace(["-"],0.0,inplace = True)

race_city.replace(["(X)"],0.0, inplace = True)

race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]] = race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

area_list = list(race_city["Geographic area"].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = race_city[race_city["Geographic area"] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

    

# Visualization:

f,ax =  plt.subplots(figsize = (12,15))

sns.barplot(x = share_white,y = area_list, color = "green",alpha = 0.5, label="White American")

sns.barplot(x = share_black, y = area_list, color = "blue", alpha = 0.7, label = "African American" )

sns.barplot(x = share_native_american, y = area_list, color = "red",alpha = 0.6, label = "Native American")

sns.barplot(x = share_asian, y = area_list, color = "yellow", alpha = 0.6, label = "Asian")

sns.barplot(x = share_hispanic, y = area_list, color = "cyan", alpha = 0.6, label = "Hispanic")

ax.legend(loc = "upper right", frameon = True) # legends visibility

ax.set(xlabel = "Percentage of Races", ylabel = "States", title = "Percentage of State's Population According to Races")

plt.show()



    

    

    

    
# datas to normalize

sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_df["Complete Ratio"] = sorted_df["Complete Ratio"]/max(sorted_df["Complete Ratio"])

data = pd.concat([sorted_data,sorted_df["Complete Ratio"]], axis = 1)

data.sort_values("area_poverty_ratio",inplace = True)



# Visualize 

f,ax1 = plt.subplots(figsize = (20,10))

sns.pointplot(x = "area_list", y = "area_poverty_ratio", data = data, color = "red", alpha = 0.8)

sns.pointplot(x = "area_list", y = "Complete Ratio", data = data , color = "lime", alpha = 0.8)

plt.text(40,0.6,"Poverty Rate",color = "red", fontsize = 17, style = "italic")

plt.text(37,0.55, "High school Complete Ratio", color = "lime", fontsize = 17, style = "italic")

plt.xlabel("States", fontsize = 15, color = "black")

plt.ylabel("Ratio", fontsize = 15, color = "black")

plt.title("High School Graduate  VS  Poverty Rate", fontsize = 20, color = "black")

plt.grid()

plt.show()
concat = pd.concat([data,df], axis = 1)

data.head()
state = police_killings_us.state.value_counts()




kill_state = pd.DataFrame({"area_list":state[::].index,"kill":state[::].values})

kill_state["kill"] = kill_state["kill"]/max(kill_state["kill"])

sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_df["Complete Ratio"] = sorted_df["Complete Ratio"]/max(sorted_df["Complete Ratio"])



data = pd.concat([sorted_data,sorted_df["Complete Ratio"]], axis = 1)

data.sort_values("area_poverty_ratio",inplace = True)



data = data.set_index("area_list")

data = data.reset_index() 

    

kill_state_count =  list(police_killings_us["state"].value_counts())

kill_state_index = police_killings_us["state"].value_counts().index[::]

kill_state = pd.DataFrame({"state":kill_state_index, "kill_count":kill_state_count })

kill_state["kill_count"] = kill_state["kill_count"]/max(kill_state["kill_count"]) 



kill_state = kill_state.set_index("state")

kill_state = kill_state.reindex(index = data["area_list"])

kill_state = kill_state.reset_index()

data =  pd.concat([data,kill_state["kill_count"]], axis = 1)

data.head()

# Visualize 

f,ax1 = plt.subplots(figsize = (20,10))

sns.pointplot(x = "area_list", y = "area_poverty_ratio", data = data, color = "red", alpha = 0.8)

sns.pointplot(x = "area_list", y = "Complete Ratio", data = data , color = "lime", alpha = 0.8)

sns.pointplot(x = "area_list", y = "kill_count", data = data, color = "black", alpha = 0.5  )

plt.text(40,0.47,"kill ratio each states",color = "black", fontsize = 17, style = "italic")

plt.text(37,0.6,"Poverty Rate",color = "red", fontsize = 17, style = "italic")

plt.text(37,0.55, "High school Complete Ratio", color = "lime", fontsize = 17, style = "italic")

plt.xlabel("States", fontsize = 15, color = "black")

plt.ylabel("Ratio", fontsize = 15, color = "black")

plt.title("High School Graduate - Poverty Rate - Kill Ratio", fontsize = 20, color = "black")

plt.grid()

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g =  sns.jointplot(data.area_poverty_ratio, data["Complete Ratio"], kind = "kde", size = 7 )



plt.savefig("grapgh.png")

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

sns.set(style="ticks")

sns.jointplot("area_poverty_ratio","Complete Ratio",data = data, size = 5, ratio = 3,  color="#4CB391")

plt.show()
police_killings_us.manner_of_death.unique()
#Plot the orbital period with horizontal boxes

sns.boxplot(x = "race", y = "age", hue = "manner_of_death", data = police_killings_us, palette = "PRGn")

plt.show()
sns.swarmplot(x = "gender", y = "age", hue = "manner_of_death", data = police_killings_us)

plt.show()
police_killings_us.gender.unique()
sns.countplot(police_killings_us.gender)

plt.title("Gender of Killed People", color = "black",  fontsize = 15)

plt.show()
# kill weapons

armed = police_killings_us.armed.value_counts()

plt.figure(figsize = (10,7))

sns.barplot(x = armed[:7].index, y = armed[:7].values)

plt.title("Kill Weapons", color = "black", fontsize = "15")
police_killings_us.head()
# Age of killed people

above_below = [ "Above" if i >=25 else "Below" for i in police_killings_us.age]

df = pd.DataFrame({"Age": above_below})

sns.countplot(x = df.Age)

plt.title("Above and Below of killed people 25 age ", color = "green", fontsize = 15) 
# Race of Killed People

sns.countplot(data = police_killings_us, x = "race")

plt.title("Race of Killed People")
sns.countplot(police_killings_us.flee)

plt.xlabel("Flee Types")

plt.ylabel("Count")

plt.title("Flee Types", color = "green", fontsize = 15)
# Having body cameras or not for police

sns.countplot(police_killings_us.body_camera)

plt.title("Having body cameras or not for police",color ="black",fontsize=15)

plt.show()
# Most deaths by police in cities

city = police_killings_us.state.value_counts()



sns.barplot(x = city[:12].index, y = city[:12].values)

plt.title("Most Deaths By Police In Cities")
kill_state = pd.DataFrame({"area_list":city[::].index,"kill":city[::].values})

kill_state["kill"] = kill_state["kill"]/max(kill_state["kill"])

sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_df["Complete Ratio"] = sorted_df["Complete Ratio"]/max(sorted_df["Complete Ratio"])



data = pd.concat([sorted_data,sorted_df["Complete Ratio"]], axis = 1)

data.sort_values("area_poverty_ratio",inplace = True)



data = data.set_index("area_list")

data = data.reset_index() 

    

kill_state_count =  list(police_killings_us["state"].value_counts())

kill_state_index = police_killings_us["state"].value_counts().index[::]

kill_state = pd.DataFrame({"state":kill_state_index, "kill_count":kill_state_count })

kill_state["kill_count"] = kill_state["kill_count"]/max(kill_state["kill_count"]) 



kill_state = kill_state.set_index("state")

kill_state = kill_state.reindex(index = data["area_list"])

kill_state = kill_state.reset_index()

data =  pd.concat([data,kill_state["kill_count"]], axis = 1)

data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.set(color_codes=True)

sns.lmplot(x = "area_poverty_ratio", y ="Complete Ratio", markers = "x",data =data , height = 6)

sns.lmplot(x = "area_poverty_ratio", y ="kill_count",data =data, palette = "Set1",height = 6)



plt.show()
data.head()
sns.kdeplot(data.area_poverty_ratio,data.kill_count,shade = True, cut =3)

plt.show()
sns.kdeplot(data["Complete Ratio"],data.area_poverty_ratio,shade = True, cmap = "Reds")

plt.show()
data.head()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data = data, palette = pal, inner = "points")

plt.show()
# Correlation between area poverty ratio, highschool complete ratio and kill ratio

data.corr()
# Visualizaiton

f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(data.corr(), annot = True, linewidths = 0.5,linecolor="black", fmt = ".1f", ax = ax)

plt.show()
data.head()
sns.pairplot(data)

plt.show()