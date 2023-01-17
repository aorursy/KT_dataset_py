# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WorldCupMatches.csv")

data.info()
data.columns
data.corr()
f,ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
# this function rewrite the given list if the entries have space between the words
def f(liste):
    liste1=[]
    for i in (liste):
        if len(i.split())==2:
            i=i.split()[0]+"_"+i.split()[1]
            
        elif len(i.split())==3:
            i=i.split()[0]+"_"+i.split()[1]+"_"+i.split()[2]
        else:
            i=i.split()[0]
        liste1.append(i)
    return liste1
        
data.columns
# use the function f on the columns of the data
# for now I couldn't find any way to write the function as list comprehension
data.columns=[f(data.columns)]
data.head()
#what is the meaning of multiindex? 
data.columns
data.describe()
data.info()
#add new column to data
#df=data.head(100)

#data1=df.set_index(["Stage","Home_Team_Goals"])

#df["Total_Goals"]=df.Home_Team_Goals + df.Away_Team_Goals

#df["Home_Team_Total_Goals"]=[df["Home_Team_Goals"]+df["Half-time_Home_Goals"]]
#data["Away_Team_Total_Goals"]=data.Away_Team_Goals+data.Half-time_Away_Goals
data2=pd.read_csv("../input/WorldCups.csv")
data2.head(10)
data2.info()
data2
data2.GoalsScored.plot(kind="line", color="r", linestyle=":",figsize=(10,10))
plt.ylabel("GoalsScored")
plt.title("Line Plot")
plt.show()


data3 = data2.set_index(["Country","Winner"]) 
data3
goal_mean=sum(data2.GoalsScored)/len(data2.GoalsScored)
print(goal_mean)
data2["GoalLevel"]=["high" if i> goal_mean else "low" for i in data2.GoalsScored]

data2.head()
data2.loc[:,["GoalsScored","GoalLevel"]]
data2.MatchesPlayed.plot(kind="hist",color="b",label="MatchesPlayed")
plt.legend(loc='upper left')
plt.xlabel("MatchesPlayed")
plt.show()
data2.plot(kind='scatter', x='MatchesPlayed', y='GoalsScored',alpha = 0.5,color = 'red')
plt.xlabel('MatchesPlayed')              
plt.ylabel('GoalsScored')
plt.title('MatchesPlayed-GoalsScored Scatter Plot') 
plt.show()
data2.boxplot(column='GoalsScored',by = 'MatchesPlayed')
plt.show()
#subplots
data2.plot(grid=True, alpha=0.9,subplots=True, figsize=(10,10)) 
plt.show()
#melting
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
data_new=data2.head(10)
melted = pd.melt(frame=data_new,id_vars = 'Winner', value_vars= ['GoalsScored',"QualifiedTeams","MatchesPlayed"])
melted
#concetenating data
data5=data2.head()
data6=data2.tail()
concat_data=pd.concat([data5,data6],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
concat_data
concat_data_col1=pd.concat([data5.GoalsScored,data5.GoalLevel],axis=1), #axis=1 means concat datas by columns
concat_data_col1
df1=data2["GoalsScored"].head()
df2=data2["GoalLevel"].head()
concat_data_col2=pd.concat([df1,df2],axis=1)
concat_data_col2
data7=data2.loc[:,["GoalsScored","QualifiedTeams","MatchesPlayed"]]
data7.plot()
plt.show()
data7.plot(subplots=True)
plt.show()
#scatter plot
data7.plot(kind="scatter",x="MatchesPlayed",y="QualifiedTeams")
plt.show()
data7
#histogram plot
data7.plot(kind="hist",y="GoalsScored",bins=20,range= (0,150),figsize=(8,8),color="purple")
plt.title("GoalsScored Histogram Plot")
plt.xlabel("Goals")
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data7.plot(kind = "hist",y = "GoalsScored",bins = 20,range= (0,180),normed = True,ax = axes[0])
data7.plot(kind = "hist",y = "GoalsScored",bins = 20,range= (0,180),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
data2.head()
# histogram subplot with non cumulative and cumulative
fig,axes=plt.subplots(nrows=2,ncols=1)
data5.plot(kind="hist",y="MatchesPlayed",bins=20,ax=axes[0])
data5.plot(kind="hist",y="MatchesPlayed",bins=20,ax=axes[1],cumulative=True)
plt.savefig('graph1.png')
plt.show()
data2.head()
data2["GoalsScored"]
# indexing using square brackets
data2["GoalsScored"][0]
# using column attribute and row label
data2.GoalsScored[2]
# using loc accessor
data2.loc[0,["GoalsScored"]]
# using loc accessor
data2.loc[4,["GoalsScored"]]
# Selecting only some columns
data2[["GoalsScored","QualifiedTeams","Attendance"]]

# or we can use this method to get some columns
data7=data2.loc[:,["QualifiedTeams","GoalsScored","MatchesPlayed"]]
data7
data2.head()
# Slicing and indexing series
data2.loc[0:10,"GoalsScored":"Attendance"] 
# Reverse slicing 
data2.loc[10:0:-1,"GoalsScored":"Attendance"] 
# From something to end
data2.loc[0:10,"GoalsScored":] 
# From beginning to something
data2.loc[0:10,:"Fourth"] 
data2
#filtering data frames
boolean=data2.GoalsScored > 100
data2[boolean]
# Combining filters
first_filter=data2.GoalsScored >80
second_filter=data2.MatchesPlayed >40
data2[first_filter & second_filter]
data2[data2.GoalsScored <90]
data2.GoalsScored[data2.QualifiedTeams < 16]
# transfroming data and list comprehension
data2["GoalMean"]=[round(data2.GoalsScored[i]/data2.MatchesPlayed[i],2) for i in range(len(data2.GoalsScored))]
data2
#index object
print(data2.index.name)
data2.index.name="IndexName"
data2.head()
# Overwrite index
# if we want to modify index we need to change all of them.
# first copy of our data to data3 then change index 
data8 = data2.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data8.index = range(1,21,1)
data8
# Setting index : type 1 is outer type 2 is inner index
data9 = data2.set_index(["Winner","Year"]) 
data9
data10=data2.set_index(["Winner","GoalMean"])
data10
data2.describe()
data2.groupby("GoalsScored").mean()
data2.groupby("QualifiedTeams").GoalMean.min()
data2.groupby("MatchesPlayed")[["GoalMean","Attendance"]].max()
