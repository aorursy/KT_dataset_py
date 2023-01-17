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
#data.head()
data = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

data.info()
data[["home_team","away_team","tournament","city","country"]] = data[["home_team","away_team","tournament","city","country"]].apply(lambda x : x.astype("category"))

data["date"] = pd.to_datetime(data["date"])
data.info()
tour_kind = list(data.tournament.unique())
tour_goal_rate = []

for i in tour_kind:

    t = data[data["tournament"] == i]

    rate = (sum(t.home_score)+sum(t.away_score))/len(t)

    tour_goal_rate.append(rate)



dictionary = {"tour_goal_rate" : tour_goal_rate, "tour_kind" : tour_kind}

new_data = pd.DataFrame(dictionary)

new_index  = (new_data["tour_goal_rate"].sort_values(ascending = False)).index.values

sorted_data = new_data.reindex(new_index)
sorted_data.index
plt.figure(figsize = (25,10))

sns.barplot(x = sorted_data["tour_kind"],y = sorted_data["tour_goal_rate"])

plt.xticks(rotation = 90)

plt.xlabel = ("soccer tour kind")

plt.ylabel("soccer tour goal rate")

plt.title("goal rate and tournament kind")

plt.show()
data.head()



c_list = list(data.country.unique())

c_list
total_goal = []

for i in c_list:

    c = data[data["country"]==i]

    sum_goal  =  sum(c.home_score)+sum(c.away_score)

    total_goal.append(sum_goal)

    

dictionary = {"country_list" : c_list, "total_goall" : total_goal}

new_data = pd.DataFrame(dictionary)

new_index = (new_data["total_goall"].sort_values(ascending = True)).index.values

sorted_data2 = new_data.reindex(new_index)
x = sorted_data2["country_list"][:15]

y = sorted_data2["total_goall"][:15]

len(x)
plt.figure(figsize = (15,10))

sns.barplot(x = x,y=y,palette=sns.cubehelix_palette(len(x)))

plt.xticks(rotation =90 )

plt.title("total goal each country")

plt.show()
data.head()
home_tlist = list(data.home_team.unique())[:20]

home = []

away = []

for i in home_tlist:

    tl = data[data.home_team == i]

    count_home = sum(tl.home_score)/len(tl)

    count_away = sum(tl.away_score)/len(tl)

    home.append(count_home)

    away.append(count_away)
sözlük = {"home_goal":home,"team_list":home_tlist}

sözlük2 = {"away_goal":away,"team_list":home_tlist}

n_data1 = pd.DataFrame(sözlük)

n_data2 = pd.DataFrame(sözlük2)

index1 = n_data1["home_goal"].sort_values(ascending = False).index.values

index2 = n_data2["away_goal"].sort_values(ascending = False).index.values

sorted_data3 = n_data1.reindex(index1)

sorted_data4 = n_data2.reindex(index2)
f,ax = plt.subplots(figsize = (10,10))

sns.barplot(x =sorted_data3.home_goal,y = sorted_data3.team_list,color = "pink",alpha = .5,label = "home_score_rate")

sns.barplot(x = sorted_data4.away_goal,y =sorted_data4.team_list,color = "purple",alpha = .6,label = "away_score_rate")

plt.xticks(rotation = 90)

ax.legend(loc = "upper_right",frameon = True)

ax.set(xlabel = "country",ylabel = "rate",title = "country home and away goal rate")
f,ax = plt.subplots(figsize = (10,10))

sns.pointplot(x = 'team_list',y = 'away_goal',data = sorted_data4,color = "green",alpha = .6)

sns.pointplot(x = 'team_list',y = 'home_goal',data = sorted_data3,color = "blue",alpha = .7)

plt.xticks(rotation = 90)

plt.text(1,1,"home goal rate",color = "green",fontsize  = 13,style = "italic")

plt.text(1,2,"away goal rate",color = "blue",fontsize = 13, style = "italic")

plt.grid()

plt.show()

sns.jointplot(sorted_data3["home_goal"],sorted_data4["away_goal"],kind = "kde",size = 8,alpha = .3,color = "purple")

plt.savefig("home and awy goal")

plt.show()
data.head()

data.country.dropna()

labels = data.country.value_counts()[:6].index

explode = [0,0,0,0,0,0]

sizes = data.country.value_counts()[:6].values

colors = ["pink","purple","blue","lime","red","green"]

plt.figure(figsize = (7,7))

plt.pie(sizes,explode = explode ,labels = labels,colors = colors,autopct="%1.1f%%")

plt.title("first six country in tournament ",color = "red",fontsize = 15)

plt.show()
sorted_data3 

sorted_data4
data5 = pd.concat([sorted_data3,sorted_data4["away_goal"]],axis = 1)
data5.index = data5.team_list
#data5.drop(columns = ["team_list"],inplace = True)
sns.lmplot(x = "home_goal",y = "away_goal",data = data5)

data5.corr()
sns.kdeplot(data5.home_goal,data5.away_goal,shade = True , cut = 4)

plt.show()
pal = sns.palplot(sns.color_palette("Set1"))


sns.violinplot(data = data5,palette =pal , inner = "points",width = .6,saturation = 170)

plt.show()
f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(data5.corr(),annot = True,linewidths= .6,linecolor="orange",fmt = ".1f",ax=ax)

plt.show()
sns.pairplot(data5)
sns.countplot(data.neutral)
data.head()
total = data.home_score+data.away_score

df = pd.DataFrame({"total_goal":total})

top_under = ["top" if i>=3 else "under" for i in df.total_goal]


top_under = pd.DataFrame({"top":top_under})

sns.countplot(x = top_under.top)

plt.show()

goal = ["yes_goal" if i >= 1 else "no_goal" for i in data.home_score]

home_team = pd.DataFrame({"goal":goal})
goal1 = ["yes_goal" if i >= 1 else "no_goal" for i in data.away_score]

away_team = pd.DataFrame({"goal1":goal1})
plt.subplots(figsize = (5,5))

sns.countplot(away_team.goal1)

sns.countplot(home_team.goal)
sns.swarmplot(x = data5.index,y = data5.home_goal)

plt.xticks(rotation = 90)