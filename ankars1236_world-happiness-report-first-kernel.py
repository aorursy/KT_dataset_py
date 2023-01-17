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
data = pd.read_csv ("../input/2017.csv") # reading csv files for years 2017, 2016 and 2015
data2 = pd.read_csv ("../input/2016.csv")
data3 = pd.read_csv ("../input/2015.csv")
data.info()       # looking general info about data
data2.info()
data3.info ()

data.describe()   # main descriptives about data of 2017

data2.describe ()            # main descriptives about data of 2016
data3.describe ()        # main descriptives about data of 2015
data.head()    # top 5 countries in 2017

data2.head ()            # top 5 countries in 2016
data3.head ()                # top 5 countries in 2015
data.tail ()          #last 5 countries in 2017
data2.tail()            #last 5 countries in 2016
data3.tail ()                #last 5 countries in 2015
data.corr ()        #correlations between variables in order to understand data
#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
# Scatter Plot 
# x = Economy..GDP.per.Capita, y = Health..Life.Expectancy
data.plot(kind='scatter', x='Economy..GDP.per.Capita.', y='Health..Life.Expectancy.',alpha = 0.9,color = 'green')
plt.xlabel('Economy..GDP.per.Capita.')              # label = name of label
plt.ylabel('Health..Life.Expectancy.')
plt.title('GDP vs Life Expectancy Scatter Plot')            # title = title of plot
plt.show ()
# Scatter Plot 
# x = Happiness.Score, y = Freedom
data.plot(kind='scatter', x='Happiness.Score', y='Freedom',alpha = 0.9,color = 'blue')
plt.xlabel('Happiness.Score')              # label = name of label
plt.ylabel('Freedom')
plt.title('Happiness.Score vs Freedom Scatter Plot')            # title = title of plot
plt.show ()
# Histogram
# bins = number of bar in figure
data.Freedom.plot(kind = 'hist',bins = 30,figsize = (11,11))
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Generosity.plot(kind = 'line', color = 'g',label = 'Generosity',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
data.Freedom.plot(kind = 'line', color = 'r',label = 'Freedom',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
data.Family.plot (kind = 'line', color = 'b', label = 'Family', linewidth = 1, alpha = 0.9, grid = True, linestyle = '--')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data = pd.read_csv ("../input/2017.csv")
x = data['Happiness.Rank']<= 20     # top 20 countries in happiness rank in 2017
data[x]

y = data2['Happiness Rank']<= 20     # top 20 countries in happiness rank in 2016
data2[y]
z = data3['Happiness Rank']<= 20     # top 20 countries in happiness rank in 2015
data3[z]
data = pd.read_csv ("../input/2017.csv")
data.columns = data.columns.str.replace(".","_") # changes the columns' names which contain "."
data.columns
data["happy_or_not"] = ["happy" if i < 20 else "not_happy" for i in data.Happiness_Rank] 
# adding a new column "happy_or_not" to see the countries who are happy
data.loc[:,["Country","happy_or_not"]] #gives the info of all country names and their happiness
data2017 = data.loc[:19,["Country"]]
data2017 = data2017.rename(columns={'Country': '2017'})
data2016 = data2.loc [:19,["Country"]]
data2016 = data2016.rename(columns={'Country': '2016'})
data2015 = data3.loc [:19, ["Country"]]
data2015 = data2015.rename(columns={'Country': '2015'})
first20 = pd.concat ([data2017,data2016,data2015] , axis = 1)

first20  # for 2017,2016 and 2015 , first 20 countries
