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
data = pd.read_csv("../input/2017.csv")
data.info()
data.columns
data.head(10) #shows first 10 rows
data.rename(columns={"Economy..GDP.per.Capita.":"GDP"}, inplace = True) 
data.rename(columns={"Happiness.Rank":"HappinessRank"}, inplace= True)
data.tail(10) #shows last 10 rows
data.corr() #creates a correlation figure with rates among the columns 
#Correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.01f',ax=ax)

plt.show()
#histogram

#bins = number of bar in figure

data.GDP.plot(kind="hist",bins=154,figsize=(15,15))

plt.show()
ax = plt.gca()



data.plot(kind='line', x = "HappinessRank",y = "GDP", color = "green", ax=ax,grid = True,figsize = (15,15))

data.plot(kind='line', x = "HappinessRank",y = "Freedom", color = 'red', ax=ax,grid = True)

data.plot(kind='line', x = "HappinessRank",y = "Trust..Government.Corruption.", color = 'b', ax=ax,grid = True)

plt.legend(loc = "upper left")

plt.show()
data.rename(columns={"Happiness.Score":"HappinessScore"}, inplace = True) 
#Let's classify countries whether they are livable or not. Our treshold is is average happiness score

#(Of course that is not the right way to determine countries that livable or not)  

threshold = sum(data.HappinessScore)/len(data.HappinessScore)

data["livable"] = [True if i > threshold else False for i in data.HappinessScore]

data.loc[::5,["Country","GDP","livable"]]
data.describe()
#Black line at top is max

#Blue line at top is 75%

#Blue (or middle) line is median (50%)

#Blue line at bottom is 25%

#Black line at bottom is min

data.boxplot(column="HappinessScore", by="livable")