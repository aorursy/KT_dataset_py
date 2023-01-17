# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data.describe()
data.head()
data.tail()
data.corr()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
data.plot(figsize=(10,10))
data.budget.plot(kind = 'line', color = 'g',label = 'budget',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
#data.revenue.plot(color = 'r',label = 'revenue',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
#data.budget.plot(kind = 'line', color = 'g',label = 'budget',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
data.revenue.plot(color = 'r',label = 'revenue',linewidth=1, alpha = 0.5,grid = True,linestyle = '-',figsize=(10,10))
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# budget-revenue plot for first 10 items
plt.subplots(figsize=(10,10))
plt.plot(data.budget[0:9],linestyle=":")
plt.plot(data.revenue[0:9],linestyle="-.",color="r")

plt.title("budget-revenue plot")
plt.legend()
plt.show()
# budget-revenue plot
#data.plot(kind="line",x="runtime",y="revenue",linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(20,10))
#plt.xlabel("budget")
#plt.ylabel("revenue")
#plt.title("budget-revenue plot")
#plt.show()
# runtime-popularity report
data.plot(kind="line",color="r",x="runtime",y="popularity",alpha = 0.5,figsize=(20,10))
plt.xlabel("runtime")
plt.ylabel("popularity")
plt.title("runtime-popularity report")
plt.legend()
plt.show()
# popularity-runtime-vote_average plot
plt.subplots(figsize=(10,10))
plt.plot(data["popularity"][0:10],data["runtime"][0:10],color="r",alpha = 0.5, linestyle = ':')
plt.plot(data["popularity"][0:10],data["revenue"][0:10],color="g",linestyle="-.")
plt.title("popularity-runtime-revenue plot")
plt.xlabel("popularity")
plt.legend()
plt.show()
#scatter plot
plt.subplots(figsize=(10,10))
plt.scatter(data.runtime[0:49],data.budget[0:49],color="g",alpha=0.5)
plt.grid()
#histogram1
data.revenue.plot(kind="hist",bins=10,figsize=(10,5))
plt.title("Revenue Histogram")
plt.xlabel("revenue")
plt.show()
#histogram2
data.popularity.plot(kind="hist",bins=10,figsize=(10,5),color="r")
plt.title("Popularity Histogram")
plt.xlabel("popularity")
plt.show()
#histogram3
data.runtime.plot(kind="hist",bins=10,figsize=(10,5),color="purple")
plt.title("Runtime Histogram")
plt.xlabel("runtime")
plt.show()
#subplots
budget=data["budget"]
revenue=data["revenue"]
popularity=data["popularity"]

plt.subplot(3,1,1)
plt.title("budget-revenue-popularity graph")
plt.plot(budget,color="r",label="budget")
plt.grid()

plt.legend()

plt.subplot(3,1,2)
plt.plot(revenue,color="b",label="revenue")
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(popularity,color="g",label="popularity")
plt.grid()
plt.legend()
plt.show()

# automaticly gives subplot.
data.plot(grid=True, alpha=0.9,subplots=True, figsize=(10,10)) 
plt.show()
data.production_countries
production_countries.head()
#bar plot1
production_countries=data["production_countries"]
plt.bar(production_countries[0:10],budget[0:10],color="purple",align='center')
#plt.figure(figsize = [10,10])
plt.xlabel("production_countries")
plt.ylabel("budget")
plt.title("production_countries-budget bar plot")
plt.show()

#bar plot
production_companies=data["production_companies"]
runtime=data["runtime"]
plt.bar(production_companies[0:10],runtime[0:10],color="r")
#plt.figure(figsize = [10,10])
plt.xlabel("production_companies")
plt.title("production_companies-runtime bar plot")
plt.show()

