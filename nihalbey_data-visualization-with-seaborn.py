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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.columns
#percantage poverty level according to area

percentage_people_below_poverty_level.poverty_rate.replace("-",0.0,inplace=True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area = list(percentage_people_below_poverty_level["Geographic Area"].unique())

liste = []

for i in area:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    mean = sum(x.poverty_rate)/len(x)

    liste.append(mean)

data = pd.DataFrame({"area":area,"value":liste})

first_sorted = (data["value"].sort_values(ascending = False)).index.values

sorted_data = data.reindex(first_sorted)

#vizualiton time :)

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["area"],y=sorted_data["value"])

plt.xticks(rotation=90)

plt.xlabel("city")

plt.ylabel("poverty rate")

plt.title("poverty rate of each state")

#Most common name of killed people

separated = kill.name[kill.name != "TK TK"].str.split()

a,b = zip(*separated)

new_list = a+b

count_name = Counter(new_list)

most_common = count_name.most_common(15)

x,y = zip(*most_common)

x,y = list(x),list(y)

# vizualiton

plt.figure(figsize=(15,10))

ax = sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
percent_over_25_completed_highSchool.percent_completed_hs.replace("-",0.0,inplace=True)

area = list(percent_over_25_completed_highSchool["Geographic Area"].unique())

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

liste = []

for i in area:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    mean = sum(x.percent_completed_hs)/len(x)

    liste.append(mean)

data = pd.DataFrame({"area":area,"value_value":liste})

sorted_data1 = (data["value_value"].sort_values(ascending = False)).index.values

new_sorted_data = data.reindex(sorted_data1)

#vizualition

plt.figure(figsize=(15,10))

sns.barplot(x=new_sorted_data["area"],y=new_sorted_data["value_value"])

share_race_city.columns
#It is horizontal barplot we wanto bar plot we write indew x to y

area_last=list(share_race_city["Geographic area"].unique())

share_race_city.replace("(X)",0.0,inplace=True)

share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]]=share_race_city.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_last:

    x = share_race_city[share_race_city["Geographic area"] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

#andddd vizualition

plt.figure(figsize=(10,15))

sns.barplot(x=share_white,y=area_last,alpha = 0.5,color = "blue",label="white")

sns.barplot(x=share_black,y=area_last,alpha=0.6,color="red",label="black")

sns.barplot(x=share_native_american,y=area_last,alpha=0.5,color="lime",label="native")

sns.barplot(x=share_asian,y=area_last,alpha=0.6,color="yellow",label="asian")

sns.barplot(x=share_hispanic,y=area_last,alpha=0.7,color="cyan",label="hispanic")

plt.yticks(rotation=-45)

plt.xlabel("values")

plt.ylabel("state")

plt.title("mean of race of each state")

plt.legend(loc="lower right")
percentage_people_below_poverty_level.head()
new_sorted_data.head()
#Point plot is line plot + scatter plot

sorted_data["value"] = sorted_data["value"]/max( sorted_data["value"])

new_sorted_data["value_value"] = new_sorted_data["value_value"]/max( new_sorted_data["value_value"])

data = pd.concat([sorted_data,new_sorted_data["value_value"]],axis=1)

data.sort_values("value",inplace=True)

#vizualition

plt.figure(figsize=(20,10))

sns.pointplot(x="area",y="value",data=data,color="blue",alpha=0.6)

sns.pointplot(x="area",y="value_value",data=data,color="green",alpha=0.6)

plt.text(40,0.6,'high school graduate ratio',color='green',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='blue',fontsize = 18,style = 'italic')

plt.xticks(rotation=90)

plt.xlabel("area")

plt.ylabel("value")

plt.title("poverty rate vs high school graduation",color="red",fontsize=24)
#kde plot + violinplot = jointplot(kind = kde)

sns.jointplot(data["value"],data["value_value"],kind="kde",size=7)

plt.show()
#Scatter plot + barplot = jointpoint

sns.jointplot("value","value_value",data=data,size=5,ratio = 3,color="r")

plt.show()
kill.race.value_counts()
# Race rates according in kill data 

labels = kill.race.value_counts().index

explode = [0,0,0,0,0,0]

size = kill.race.value_counts().values

colors = ["cyan","yellow","blue","lime","red","green"]

#vizualition

plt.pie(size,explode = explode,labels = labels,colors = colors,autopct = "%1.1f%%")

plt.legend()

plt.show()
# Race rates according in kill data  with countplot

sns.set(style='darkgrid')

ax = sns.countplot(y = kill.race)
data17
# Race rates according in kill data

# we want to draw data big to smal we do so

kil_race_values = kill.race.value_counts().values

kill_race_index = kill.race.value_counts().index

data19 = pd.DataFrame({"race":kill_race_index,"count":kil_race_values})

data18 =(data19["count"].sort_values(ascending=False)).index.values

data17 = data19.reindex(data18)

sns.set(style = "darkgrid")

ax = sns.barplot(x="count",y="race",data = data17)

plt.title("The number of race in kill")
kill.columns
#There is my lmplot working maybe it is looking silly but everything for learning

#i did ı seperated kill data male or female and i draft age according to id and seperated mental illnes

ax = sns.lmplot(x = "id",y="age",data=kill,col="signs_of_mental_illness")