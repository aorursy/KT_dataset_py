# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
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
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level['Geographic Area'].unique()
percentage_people_below_poverty_level.replace(["-"],(0.0),inplace = True)
#I dont know what "-" it means so replaced the "-" values with 0
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
#I converted all (there is some int values in dataset)values to float 
area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())
#I get state list
area_poverty_ratio = [] 
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})
new_id = (data["area_poverty_ratio"].sort_values(ascending=True)).index.values
#ascending=True-->Low to high / ascending=False -->High to low
sorted_data= data.reindex(new_id)
sorted_data
#we sorted the data --> low to high.

#visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'],palette = sns.cubehelix_palette(len(x)))
#sns.cubehelix_palette --> This function creates colors
plt.xticks(rotation=45)#rotation of text in axix
plt.xlabel=("States")
plt.ylabel=("Poverty Rate")
plt.title=('Poverty Rate Given States')
plt.show()
kill.head()
kill.name.value_counts() #some missing data which name is TK TK 
seperate = kill.name[kill.name != "TK TK"].str.split()
#we clear the data and seperate the names("Burak Arslan"-->"Burak" "Arslan")
#seperate = kill.name[kill.name != "TK Tk"].str.split()
a,b = zip(*seperate)#we zipped the seperated datas
name_list = a+b 
name_count = Counter(name_list)#we get number of names
most_common_names = name_count.most_common(10)#we found the most common 10 names
x,y = zip(*most_common_names)#we zipped the data
x,y = list(x),list(y)#we created x and y 
plt.figure(figsize=(15,10))
ax = sns.barplot(x=x, y=y,palette="Blues_d")

plt.xlabel=("Name or Surname")
plt.ylabel=("Number of Name and Surname")
plt.title=("Most Common 10 Name or Surname")
plt.show()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())
area_highschool = []
for i in area_list:
    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})
new_index = (data["area_highschool_ratio"].sort_values(ascending=False)).index.values
sorted_data2 = data.reindex(new_index)
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2["area_list"],y=sorted_data2["area_highschool_ratio"])
plt.xticks(rotation = 30)
plt.xlabel=("States")
plt.ylabel=("High School Graduade Rate")
plt.title=("Percentage")
plt.show()
share_race_city.head()
share_race_city.info()
share_race_city.replace(["-"],0.0,inplace = True)
share_race_city.replace(["(X)"],0.0,inplace = True)
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())

#created a list for each of label because we have city percantages but we need state percentage 
share_white = []
share_black = []
share_native_american= []
share_asian= []
share_hispanic= []

for i in area_list:
    #we add mean of values for each state
    x=share_race_city[share_race_city["Geographic area"]==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))

#visualization
f , ax = plt.subplots(figsize = (13,20))
sns.barplot(x=share_white,y=area_list,color="#8c001a", alpha=0.9,label="Whites")#alpha = oppacity
sns.barplot(x=share_black,y=area_list, color="#00fdd1",alpha=0.9,label="Black")
sns.barplot(x=share_native_american,y=area_list, color="#2701d5",alpha=0.9,label="Native")
sns.barplot(x=share_asian,y=area_list, color="#ffd62a",alpha=0.9,label="Asian")
sns.barplot(x=share_hispanic,y=area_list, color="#46a346",alpha=0.9,label="Hispanic")
# (#46a346) Hex color codes.Not important.
ax.legend(ncol=2,loc="upper right",frameon=True)#frameon --> frame on legend
ax.set(xlabel="Percentage",ylabel="States")#we can set labels easily
sns.despine(left=True, bottom=True)
x =[1,2,3,4,5]
y =[1,2,3,4,5]
yi =[500,400,300,200,100]

plt.plot(x,y)
plt.plot(x,yi)
plt.show()

#if we normalize the dataset

x1 =[]
y1 =[]
yi1 =[]
for i in x:
    x1.append(i/max(x))
for a in y:
    y1.append(a/max(y))
for b in yi:
    yi1.append(b/max(yi))


plt.plot(x1,y1)
plt.plot(x1,yi1)
plt.show()


data_highschool = pd.DataFrame({"area_list":area_list,"area_poverty_ratio":area_poverty_ratio})
data_highschool = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

##
sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])
sorted_data2["area_highschool_ratio"] = sorted_data2["area_highschool_ratio"]/max(sorted_data2["area_highschool_ratio"])
#normalization
data = pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1)
data_highschool= pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1)

data.sort_values("area_poverty_ratio",inplace= True)
data_highschool.sort_values("area_highschool_ratio",inplace= True)
#concat and sort datas 

f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x="area_list",y="area_poverty_ratio",data=data,color="cyan")
sns.pointplot(x="area_list",y="area_highschool_ratio",data=data,color="lime")
plt.text(40,0.6,'high school graduate ratio',color='lime',fontsize = 17,style = 'italic')
plt.text(40,0.55,'poverty ratio',color='cyan',fontsize = 18,style = 'italic')
ax.set_xlabel('States',fontsize = 15,color='blue')
ax.set_ylabel('Values',fontsize = 15,color='blue')
ax.set_title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
plt.show()


f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x="area_list",y="area_poverty_ratio",data=data_highschool,color="cyan")
sns.pointplot(x=data_highschool["area_list"],y=data_highschool["area_highschool_ratio"],color="lime")
plt.text(40,0.8,'high school graduate ratio',color='lime',fontsize = 17,style = 'italic')
plt.text(40,0.75,'poverty ratio',color='cyan',fontsize = 18,style = 'italic')
ax.set(xlabel="States",ylabel="Values",title="Sorted by High School Graduate")
plt.grid()
plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="reg", size=7)
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="resid", size=7)
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="hex", size=7)
plt.savefig("graph.png")
plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
kill.race.value_counts()
kill.race.dropna(inplace= True)
labels = kill.race.value_counts().index
sizes = kill.race.value_counts().values
explode = [0,0,0,0,0,0]
colors = ["gray","cyan","red","lime","green","brown"]

plt.figure(figsize = (7,7))
plt.pie (sizes , explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# Show the results of a linear regression within each dataset
sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)
plt.show()
# Show each distribution with both violins and points
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)#colors
plt.subplots(figsize=(10,10))
sns.violinplot(data=data,palette=pal,inner="points")
plt.show()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidth=-5,fmt=".1f",ax=ax,linecolor="white")
plt.show()
kill.loc[:,["gender","age","manner_of_death"]].head()
# Plot the orbital period with horizontal boxes
sns.boxplot(x="gender",y="age",hue="manner_of_death",data=kill,palette="PRGn")
plt.show()
kill.loc[:,["gender","age","manner_of_death"]].head()
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=kill)
plt.show()
data.head()
sns.pairplot(data)
plt.show()
kill.gender.value_counts()
sns.countplot(kill.gender)
kill.armed.value_counts()
armed= kill.armed.value_counts()
plt.figure(figsize=(15,5))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.show()
above25 = ["Above25" if i>=25 else "Below25" for i in kill.age]
df = pd.DataFrame({"age":above25})
sns.countplot(x=df.age)
plt.show()
sns.countplot(data=kill, x='race')
plt.show()
city = kill.city.value_counts()
plt.figure(figsize=(20,7))
sns.barplot(x=city[:30].index,y=city[:30].values)
plt.xticks(rotation=90)
plt.show()
state = kill.state.value_counts()
plt.figure(figsize=(20,7))
sns.barplot(x=state[:10].index,y=state[:10].values)
plt.show()
sns.countplot(kill.signs_of_mental_illness)
plt.show()
sns.countplot(kill.threat_level)
plt.show()
sns.countplot(kill.flee)
plt.show()
sns.countplot(kill.body_camera)
plt.show()
plt.subplots(figsize=(20,10))
sta = kill.state.value_counts().index
sns.barplot(x=sta,y = kill.state.value_counts().values)
plt.show()