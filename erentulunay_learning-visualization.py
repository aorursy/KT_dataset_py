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
median_house_hold_income_2015= pd.read_csv("../input/MedianHouseholdIncome2015.csv",encoding="Windows-1252")
percentage_people_below_poverty_level = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv",encoding="Windows-1252")
percent_over_25_completed_high_school= pd.read_csv("../input/PercentOver25CompletedHighSchool.csv",encoding="Windows-1252")
police_killings=pd.read_csv("../input/PoliceKillingsUS.csv",encoding="Windows-1252")
share_race_city=pd.read_csv("../input/ShareRaceByCity.csv",encoding="Windows-1252")

percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.poverty_rate.value_counts()
#poverty rate of each state
percentage_people_below_poverty_level.poverty_rate.replace(("-",0.0),inplace=True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype("float")
area_ratio= []
area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]
    area_rate= x.poverty_rate.sum()/len(x)
    area_ratio.append(area_rate)
data=pd.DataFrame({"area_list":area_list,"area_ratio":area_ratio})
new_index=(data["area_ratio"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)
print(sorted_data)
x= percentage_people_below_poverty_level.groupby(by="Geographic Area").poverty_rate.mean().reset_index().sort_values(by="poverty_rate",ascending=False)
x.plot(kind="bar",x="Geographic Area",y="poverty_rate",figsize=(15,10))
plt.xlabel("Geographic Area")
plt.ylabel("poverty_rate")
plt.show()
plt.figure(figsize=(15,10))
sns.barplot(x=x["Geographic Area"],y=x["poverty_rate"],palette=sns.cubehelix_palette(len(x["Geographic Area"]),reverse=True))
plt.xticks(rotation=45)
plt.xlabel("states")
plt.ylabel("poverty_rate")
plt.title("Poverty Rate Given States")
seperate = police_killings.name[(police_killings.name != "TK TK")].str.split()
a,b=zip(*seperate)
name_list=a+b
name_counter= Counter(name_list).most_common(15)
x,y=zip(*name_counter)
x,y=list(x),list(y)
print(x)
print(y)
a= police_killings.name[police_killings.name!="TK TK"].str.split()
a,b=zip(*a)
a=pd.Series(a)
b=pd.Series(b)
x = pd.concat((a,b),axis=0)
x = x.value_counts().head(15).reset_index()
x.rename({"index":"common_names",0:"Value"},axis=1,inplace=True)
print(x)
plt.figure(figsize=(15,10))
sns.barplot(x=x.common_names,y=x.Value,palette=sns.cubehelix_palette(len(x.common_names)))
percent_over_25_completed_high_school.head()
percent_over_25_completed_high_school.percent_completed_hs.value_counts()
percent_over_25_completed_high_school.percent_completed_hs.replace("-",0.0)
percent_over_25_completed_high_school.percent_completed_hs.value_counts()
percent_over_25_completed_high_school.percent_completed_hs.replace(("-",0.0),inplace=True)
percent_over_25_completed_high_school.percent_completed_hs=percent_over_25_completed_high_school.percent_completed_hs.astype("float")
empty_list=list()
liste=list(percent_over_25_completed_high_school["Geographic Area"].unique())
for i in liste:
    total = percent_over_25_completed_high_school[percent_over_25_completed_high_school["Geographic Area"] == i]
    area_mean=sum(total.percent_completed_hs)/len(total.percent_completed_hs)
    empty_list.append(area_mean)
data_percent_b=pd.DataFrame({"Area":liste,"Mean":empty_list})
data_percent_in=(data_percent_b["Mean"].sort_values()).index.values
new_data=data_percent_b.reindex(data_percent_in)
a= percent_over_25_completed_high_school.groupby(by="Geographic Area",as_index=False).percent_completed_hs
a= a.mean().sort_values(by="percent_completed_hs")
plt.figure(figsize=(15,10))
sns.barplot(x=a["Geographic Area"] ,y=a["percent_completed_hs"],palette=sns.cubehelix_palette(len(a)))
plt.xticks(rotation=90)
plt.show()
share_race_city=pd.read_csv("../input/ShareRaceByCity.csv",encoding="Windows-1252")
share_race_city=share_race_city.replace("(X)", np.nan)
share_race_city.dropna(how="any",inplace=True)
share_race_city[["share_white","share_black","share_native_american","share_asian","share_hispanic"]]=share_race_city.loc[:,"share_white":].astype(float)
print(share_race_city.loc[:,"share_white":].dtypes)
share_white = share_race_city.groupby(by="Geographic area").share_white.mean().to_frame().reset_index()
share_black = share_race_city.groupby(by="Geographic area").share_black.mean().to_frame().reset_index()
share_native_american = share_race_city.groupby(by="Geographic area").share_native_american.mean().to_frame().reset_index()
share_asian = share_race_city.groupby(by="Geographic area").share_asian.mean().to_frame().reset_index()
share_hispanic = share_race_city.groupby(by="Geographic area").share_hispanic.mean().to_frame().reset_index()
print(share_white)
f,ax=plt.subplots(figsize=(9,15))
sns.barplot(x=share_white["share_white"],y=share_white["Geographic area"],alpha=0.4,color="g",label="share_white")
sns.barplot(x=share_black["share_black"],y=share_white["Geographic area"],alpha=0.7,color="r",label="share_black")
sns.barplot(x=share_native_american["share_native_american"],y=share_white["Geographic area"],alpha=0.7,color="cyan",label="share_native_american")
sns.barplot(x=share_asian["share_asian"],y=share_white["Geographic area"],alpha=0.7,color="y",label="share_asian")
sns.barplot(x=share_hispanic["share_hispanic"],y=share_white["Geographic area"],alpha=0.7,color="blue",label="share_hispanic")
ax.legend(loc="lower right",frameon=True)
ax.set(xlabel="race")
plt.show()
a.drop(columns="Geographic Area",axis=1,inplace=True)
poverty_percent=pd.concat((x,a),axis=1)
poverty_percent
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
a=poverty_percent.iloc[:,1:]
poverty_percent.iloc[:,1:]= sc.fit_transform(a)
print(poverty_percent)
f,ax=plt.subplots(figsize=(20,10))
sns.pointplot(x="Geographic Area",y="poverty_rate",data=poverty_percent,alpha=0.8,grid=True)
sns.pointplot(x="Geographic Area",y="percent_completed_hs",data=poverty_percent,color="r",alpha=0.8,grid=True)
ax.text(x=40,y=0.6,s="High School",fontsize=15,style="italic")
plt.grid()
sns.jointplot(x="poverty_rate",y="percent_completed_hs",data=poverty_percent,kind="kde",size=7)
plt.show()
sns.jointplot(x="poverty_rate",y="percent_completed_hs",data=poverty_percent,ratio=3,size=7)
police_killings.dropna(inplace=True)
police_killings.race.value_counts()
labels=police_killings.race.value_counts().index.values
size=police_killings.race.value_counts().values
colors=["grey","blue","red","yellow","green","brown"]
f,ax=plt.subplots(figsize=(7,7))
plt.pie(x=size,labels=labels,colors=colors,autopct="%.1f%%",startangle=90)
sns.lmplot(x="poverty_rate",y="percent_completed_hs",data=poverty_percent)
sns.kdeplot(data=poverty_percent["poverty_rate"],data2=poverty_percent["percent_completed_hs"],shade=True,size=15,cut=0.5)
f,ax=plt.subplots(figsize=(10,10))
pal = sns.cubehelix_palette(rot=-.5, dark=.3,light=1,gamma=3)
sns.violinplot(data=poverty_percent,palette=pal,inner="boxplot",ax=ax)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(poverty_percent.corr(),annot=True)
police_killings.head()
sns.boxplot(data=police_killings,x="gender",y="age",hue="manner_of_death")
sns.swarmplot(data=police_killings,x="gender",y="age",hue="manner_of_death")
sns.pairplot(poverty_percent)
share_race_city.head()
sns.countplot(share_race_city["Geographic area"])