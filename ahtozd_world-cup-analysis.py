# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WorldCups.csv")
data.info()
data.rename(columns={"Runners-Up":"RunnersUp"}, inplace = True)
data
for i in range(len(data.Attendance)):

    data.Attendance[i] = data.Attendance[i].replace(".","")

    
data.Attendance=pd.to_numeric(data.Attendance)



data.info()
data.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.show()
data.plot(kind='scatter', x='Year', y="GoalsScored",alpha = .8,color = 'blue',figsize= (6,6))

plt.legend()

plt.xlabel('Year')             

plt.ylabel("GoalsScored")

plt.title('Scatter Plot') 

plt.show()
data.columns
data["AverageGoal"] = data.GoalsScored/data.MatchesPlayed

data.plot(kind='scatter', x='Year', y="AverageGoal",alpha = .8,color = 'blue',figsize= (6,6))

plt.legend()

plt.xlabel('Year')             

plt.ylabel("AverageGoal")

plt.title('Scatter Plot') 

plt.show()
ax = plt.gca()



data.plot(kind='line', x = "Year",y = "GoalsScored", color = "green", ax=ax,grid = True,figsize = (7,7))

data.plot(kind='line', x = "Year",y = "MatchesPlayed", color = 'red', ax=ax,grid = True)

data.plot(kind='line', x = "Year",y = "QualifiedTeams", color = 'b', ax=ax,grid = True)

plt.legend(loc = "upper left")

plt.show()
pd.Series(data.Winner).replace("Germany FR","Germany").value_counts().plot('bar', grid = True)

plt.show()
first_four = data[(data.Country == data.RunnersUp) | 

             (data.Country == data.Third) | 

             (data.Country == data.Fourth)]

first = data[data.Country == data.Winner]

values = [len(first_four),len(first),len(data)-len(first_four)-len(first)]



explode = (0,0.1, 0)

labels = ["Ratio of Home First Four",'Ratio of Home-Winners', 'Ratio of Away-Winners']

colors = ['navajowhite', '#e29589',"lightsteelblue"]

plt.pie(values, labels=labels, colors=colors, startangle=0, autopct='%.1f%%',explode=explode, 

shadow= True, pctdistance = 0.5, radius = 2)

plt.savefig('foo.png')

plt.show()



        




        