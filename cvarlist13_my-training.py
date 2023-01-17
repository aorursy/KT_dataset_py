# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sbs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/NBA_player_of_the_week.csv")
type(data)
print(data.columns)
print(data.info())
print(data.dtypes)
data.describe()
data.head(15)
data.tail()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sbs.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)
plt.show()
print(data.Player)
data.columns
data.columns = [ each.lower() for each in data.columns]
data.columns = [each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in data.columns]
data.columns
filter1 = data.draft_year>1990
filter_data = data[filter1]
print(filter_data)

filter2 = data.seasons_in>5
data[filter1 & filter2]

x = data.seasons_in.mean()
print(x)                    #Average_seasons_in player of the week
data.draft_year.plot(kind = "line", color = "g",label = "Draft Year",linewidth=1,alpha = 0.5, grid = True, linestyle = ":" )
data.season_short.plot(color = 'r',label = 'Season Short',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
boston = data[data.team == "Boston Celtics"]
print(boston.describe())
lakers = data[data.team == "Los Angeles Lakers"]
print(lakers.describe())
houston = data[data.team == "Houston Rockets"]
print(houston.describe())
plt.plot(boston.seasons_in , boston.season_short, color = "green", label = "boston")
plt.plot(houston.seasons_in , houston.season_short, color = "red", label = "houston")
plt.plot(lakers.seasons_in , lakers.season_short, color = "yellow", label = "lakers")
plt.xlabel("Season In")
plt.ylabel("Season Short")
plt.legend()
plt.show()


houston.plot(kind = "line", color = "r",label = "Houston",linewidth=1,alpha = 1, grid = True, linestyle = ":" )
boston.plot(color = 'g',label = 'Boston',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind = "scatter", x = "draft_year", y = "age", alpha = 0.5, color = "red")
plt.xlabel("Draft Yearr")
plt.ylabel("Age")
plt.title("Draft Year-Age Scatter Plot")
plt.show()
data.describe()

data.plot(kind = "scatter", x = "seasons_in", y = "age", alpha = 0.5, color = "red")
plt.xlabel("season in")
plt.ylabel("Age")
plt.title("Season In-Age Scatter Plot")
plt.show()
names = data.player.value_counts()
names[names > 1]
qwe = data.team.value_counts()
qwe[qwe > 1]
names.plot(kind = "hist", bins = 100, figsize = (15,15))
plt.xlabel("Player")
plt.show()
data.age.plot(kind = "hist", bins = 100, figsize = (15,15))
plt.xlabel("Age")
plt.show()
a = data["seasons_in"]>10
data[a]
b = data["age"]>data.age.mean()
data[b]    #older average age player
data[np.logical_and(data["seasons_in"]>5, data["age"]>26)]
for index, value in data[["age"]][0:15].iterrows():
    print(index," : ",value)
def tuble_ex():
    t=data.weight.head()
    return t
a, b, c, d, e = tuble_ex()
print(a, b, c, d, e)
    
    
x = (data.weight.head(3))      #global scope
def f():
    y = x.append(data.weight.tail(3))      #local scope
    return y
print(x)
print(f())
def square():
    """return square of value"""
    def add():
        """add to local variable"""
        a = data.age.head(5)
        b = data.age.tail(5)
        c = sum(a) + sum(b)
        return c
    return add()**2
print(square())
        
        
o = data.weight.head(1)
u = int(o)
print(u)

def f(i, j=u, k=1):
    m = i + j + k
    return m
print(f(3,))
print(f(1,2,3))
def f(*args):
    for i in args:
        print(i)
print("")
f(u,2,3)  #u is data.weight.head(1)


def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = 'usa', city = 'los angles', team = "L.A Lakes")   #kwargs is dictionary, args is list
        
square = lambda x : x**2
print(square(int(data.weight.head(1))))  #labmda function for square

number_list = [data.age.tail(3)]
y = map(lambda x:x**2,number_list)     #map func use
                                       #y = list
print(list(y))


name = "michael"
print(name)
it = iter(name)
print(next(it))
print(*it)
n_j =data[(data.seasons_in >= 1) & (data.team == "New Jersey Nets")]
print(n_j) #new jersey nets team members seasons in mvp
num1 = [data.age.head()]
num2 = [i + 1 for i in num1]
print(num1)
print(num2)
data.columns
data[["team", "player"]]

ort_yas = data.age.mean()
print(ort_yas)
data["age_average"] = ["old" if i > ort_yas else "young" for i in data.age]
data.loc[::,["age_average","age","team"]]



print(data['player'].value_counts(dropna = False))
data_new = data.head(10)
melted = pd.melt(frame=data_new,id_vars = "team", value_vars = ["age", "player"])
melted

#CONCATENATING DATA
data11 = data.head()
data12 = data.tail()
conc_data_row = pd.concat([data11,data12],axis=0,ignore_index=True) #vertical
conc_data_row
data22 = data["player"].head()
data33 = data["team"].head()
conc_data_col = pd.concat([data22,data33],axis=1)  #horizontal concat
conc_data_col
data.dtypes
data["conference"] = data["conference"].astype("category")
data["age"] = data["age"].astype("float")
data.dtypes  #change types feature 
data.info()  #conference columns is missing data
print(data['conference'].value_counts(dropna = False))
data_new = data
data_new["conference"].dropna(inplace=True)
assert data_new["conference"].notnull().all() #is true
data_new["conference"].fillna("empty",inplace=True) 
#BUILDING DATA FRAME FROMS SCRATCH
teamm = ["houston","boston"]
playerr = ["T-Mac","Garnet"]
list_label = ["teamm","playerr"]
list_col = [teamm,playerr]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["color"] = ["red","green"]
df
df["income"] = 0
df
#VISUAL EXPLORATORY DATA ANALYSIS
data1 = data.loc[:,["age","real_value","seasons_in"]] #data içinde belirtilen columns lara ait tüm bilgileri tek yerde grafike ederiz
data1.plot()
plt.show()
data1.plot(subplots=True)
data.plot(kind="scatter", x="age", y ="seasons_in")
data1.plot()
data1.plot(kind="hist", y="age", bins=50,range=(0,42),density = True)
data.describe()
data.head()
data2=data.head(100)
date_list=data.date
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2=data2.set_index("date")
data2
print(data2.loc["1985-04-07"])
print(data2.loc["1985-04-07":"1989-03-05"])
data2.resample("A").mean()  #A is year
data2.resample("M").mean()  # M is mounth
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("../input/NBA_player_of_the_week.csv")
data.head()
data["Player"][0]
data.Player[1]
data.loc[:,"Player"]
data.loc[0:10,["Player"]] #using loc accessor
data[["Player","Team"]]   ##selecting only some columns
data.loc[0:20,"Age":"Player"]  #Slicing and indexing series
data.loc[20:0:-1,"Age":"Player"] #reverse slicing
data.loc[0:10,"Player"] #from something to end
boolean = data.Age > 38
data[boolean]
filter_data1 = data.Age > 35
filter_data2 = data.Conference == "East"
data[filter_data1 & filter_data2]
data[data.Age > 35]
data.Player[data.Age > 35]   #filtering columns based others
def div(n):
    return n/2
data.Age[0:10].apply(div) #transforming data
data.Age.apply(lambda n : n/2) #another way tronsforming data
data["random_column"] = data.Age + data["Seasons in league"]
data.head()
print(data.index.name)
data.index.name = "index name"
data.head()
data1 = data.copy()
data1.index = range(100,1245,1)
data1.head()
data = pd.read_csv("../input/NBA_player_of_the_week.csv")
data.head()
data1 = data.set_index(["Player","Team"])
data1.head(10)
data2 = data.set_index(["Team","Player"])
data2
pd.melt(data[1:10],id_vars="Team",value_vars=["Age","Player"])
data.groupby("Team").mean()
data.groupby("Team").Age.mean()
data.groupby("Team").Age.max()
data.groupby("Team")[["Player","Age"]].max()
