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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percent_over_25_completed_highSchool.head()
share_race_city.head()
kill.head()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace("-",0.0,inplace=True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace("-",0.0,inplace=True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
#a = percent_over_25_completed_highSchool["percent_completed_hs"]
#data["percent_completed_hs"] = a
#data.head()
data = percentage_people_below_poverty_level.loc[:300,["poverty_rate","percent_completed_hs"]]
    # Line Plot 
data.poverty_rate.plot()
data.percent_completed_hs.plot()
plt.title("Line Plot",fontsize=15)
plt.show()
    # Scatter Plot
data.plot(kind="scatter",x="poverty_rate",y="percent_completed_hs", alpha=0.3)
plt.title("Scatter Plot",fontsize=15)
plt.xlabel("Poverty Rate",fontsize=11)
plt.ylabel("Percent Completed High School", fontsize=11)
plt.show()
    # Histogram Plot
percent_over_25_completed_highSchool.percent_completed_hs.plot(kind="hist", figsize=(13,9), bins=51, range=(1,100))
plt.ylabel("Frequency", fontsize=11)
plt.title("Histogram Plot",fontsize=15)
plt.show()
    # Subplots
data.plot(subplots=True, figsize=(13,9))
plt.title("Subplots",fontsize=15)
plt.show()
    # Bar Plot
area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())
poverty_ratio= []
for i in area_list:
     x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == i]
     rate = sum(x.poverty_rate)/len(x)
     poverty_ratio.append(rate)
data = pd.DataFrame({"Area_List":area_list,"Poverty_Ratio":poverty_ratio})
new_index = data["Poverty_Ratio"].sort_values(ascending=False).index.values
data = data.reindex(new_index)

plt.figure(figsize=(17,9))
sns.barplot(x="Area_List",y="Poverty_Ratio",data=data)
plt.ylabel("Poverty Ratio")
plt.xlabel("Area List")
plt.title("Bar Plot",fontsize=15)
plt.xticks(rotation=45)
plt.show()
sns.barplot(x=kill.state.value_counts().index[:10], y=kill.state.value_counts().values[:10])
plt.show()
kill.name.value_counts()
    # Bar Plot
seperate = kill.name[kill.name != "TK TK"].str.split()
x,y = zip(*seperate)
x,y = list(x),list(y)
name = x+y
name_counter = Counter(name)
most = name_counter.most_common(17)
x,y = zip(*most)
x,y = list(x),list(y)

plt.figure(figsize=(17,9))
sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))
plt.title("Bar Plot")
plt.show()
    # Bar Plot
area_list = percent_over_25_completed_highSchool["Geographic Area"].unique()
hs_rate = []
for i in area_list:
     x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"] == i]
     ratio = sum(x.percent_completed_hs)/len(x)
     hs_rate.append(ratio)
df = pd.DataFrame({"Area_List":area_list,"Percent_Completed_High_School":hs_rate})
new_index = df["Percent_Completed_High_School"].sort_values(ascending=True).index.values
df = df.reindex(new_index)

plt.figure(figsize=(17,9))
sns.barplot(x="Area_List", y="Percent_Completed_High_School", data=df)
plt.xlabel("Area List")
plt.ylabel("Percent Completed High School")
plt.title("Bar Plot", fontsize=15)
plt.show()
share_race_city.head()
    # Bar Plot
share_race_city.replace("-",0.0,inplace=True)
share_race_city.replace("(X)",0.0,inplace=True)
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = share_race_city["Geographic area"].unique()
white = []
black = []
asian = []
american = []
hispanic = []
for i in area_list:
     x = share_race_city[share_race_city["Geographic area"] == i]
     white.append(sum(x.share_white)/len(x))
     black.append(sum(x.share_black)/len(x))
     asian.append(sum(x.share_asian)/len(x))
     american.append(sum(x.share_native_american)/len(x))
     hispanic.append(sum(x.share_hispanic)/len(x))
f,ax = plt.subplots(figsize=(17,9))
sns.barplot(x=white, y=area_list, color="b",alpha=.5, label="White")
sns.barplot(x=black, y=area_list, color="orange",alpha=.5, label="Black")
sns.barplot(x=american, y=area_list, color="r",alpha=.5, label="American")
sns.barplot(x=asian, y=area_list, color="g",alpha=.5, label="Asian")
sns.barplot(x=hispanic, y=area_list, color="y",alpha=.5, label="Hispanic")
    
ax.legend(loc='upper right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
plt.show()
    # Bar Plot
armed = kill.armed.value_counts()

plt.figure(figsize=(17,9))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.show()
df["Percent_Completed_High_School"] = df["Percent_Completed_High_School"]/max(df["Percent_Completed_High_School"])
data["Poverty_Ratio"] = data["Poverty_Ratio"]/max(data["Poverty_Ratio"])
data = pd.concat([data,df["Percent_Completed_High_School"]],axis=1)
data.sort_values("Poverty_Ratio",inplace=True)
data.head()
    # Point Plot
plt.figure(figsize=(17,9))
sns.pointplot(x="Area_List",y="Poverty_Ratio",data=data)
sns.pointplot(x="Area_List",y="Percent_Completed_High_School", data=data, color="orange")
plt.title("Point Plot", fontsize=15)
plt.xlabel("Area List")
plt.ylabel("Percent Completed High School")
plt.show()
    # Joint Plot
sns.jointplot(data.Poverty_Ratio,data.Percent_Completed_High_School, kind="kde",size=7)
plt.title("Joint Plot", fontsize=15)
plt.xlabel("Area List")
plt.ylabel("Percent Completed High School")
plt.show()
    # Pie Chart
kill.race.dropna(inplace=True)
labels = kill.race.value_counts().index
sizes = kill.race.value_counts().values
colors = ["blue","orange","brown","green","red","grey"]
explode = [0,0,0,0,0,0]

plt.figure(figsize=(11,9))
plt.pie(sizes, explode=explode,colors=colors,labels=labels, autopct="%1.1f%%")
plt.title("Pie Chart", fontsize=15)
plt.show()
data.head()
    # Lm Plot
sns.lmplot(x="Poverty_Ratio",y="Percent_Completed_High_School", data=data)
plt.title("Lm Plot",fontsize=15)
plt.show()
    # Kde Plot
sns.kdeplot(data.Poverty_Ratio,data.Percent_Completed_High_School, cut=3, shade=True)
plt.title("Kde Plot",fontsize=15)
plt.show()
    # Vionlin Plot
pal = sns.cubehelix_palette(2, rot=.3, dark=.5)
sns.violinplot(data=data, palette=pal, inner="points")
plt.title("Violint Plot", fontsize=15)
plt.show()
data.corr()
data.corr()
    # Heat Map
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot=True,fmt=".1f" ,linewidths=.5, linecolor="red",ax=ax)
plt.title("Heat Map", fontsize=15)
plt.show()
kill.head()
kill.manner_of_death.unique()
    # Box Plot
sns.boxplot(x="gender", y="age",hue="manner_of_death", data=kill,palette="PRGn")
plt.title("Box Plot",fontsize=15)
plt.show()
    # Swarm Plot
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)
plt.title("Swarm Plot", fontsize=15)
plt.show()
    # Pair Plot
sns.pairplot(data)
plt.show()
    # Count Plot
sns.countplot(kill.gender)
plt.show()
above25 = ["above25" if i >= 25 else "below25" for i in kill.age]
age = pd.DataFrame({"Above25":above25})
sns.countplot(x=age.Above25)
plt.show()