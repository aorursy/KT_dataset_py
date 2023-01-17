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
data = pd.read_csv("../input/2017.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots()

sns.heatmap(data.corr(),annot = True, ax=ax ,fmt='.1f') # annot = show numbers on squares, fmt = how many number comes after comma

plt.show()
data.head()

data.tail()
data.columns
#i want to rename columns for easy coding

data.rename(columns = {'Happiness.Rank' : 'Happinessrank',

                       'Happiness.Score':'HappinessScore',

                       'Whisker.high':'Whiskerhigh',

                       'Whisker.low':'Whiskerlow',

                       'Economy..GDP.per.Capita.':'EconomyGDP'},inplace=True)

data.columns
data.EconomyGDP.plot(kind="line",color="red",label="Economy GDP",grid=True,alpha=0.5,linestyle=":",figsize=(9,9))

plt.legend()

data.Freedom.plot(kind="line", color="blue",label="Freedom Level",grid=True,alpha=0.5,linestyle="-.",figsize=(9,9))

plt.legend()

plt.xlabel("Happiness Rank")

plt.ylabel("Parameters")

plt.title("Relations Between Economy gdp-Happiness and Freedom Happiness")

plt.show()
#Scatter plot

data.plot(kind="scatter",x="Happinessrank",y="Family",color="red",alpha=0.5,figsize=(9,9))

plt.xlabel("Happiness Rank")

plt.ylabel("Family")

plt.show()
data.EconomyGDP.plot(kind="hist",bins=60,figsize=(9,9))

plt.title("Economy GDP Level Frequency of Countries")

plt.show()
dictionary = dict([(country,economy) for country,economy in zip(data.Country,data.EconomyGDP)]) 

# this code makes the country-economyGDP

print(dictionary)

for i in dictionary.keys():

    print(i,"=>",dictionary[i])

x= data['EconomyGDP'] > 0.5

data[x]

data[(data['Family']<1.0) & (data['Generosity']>0.3)]

# or you can use

# data(np.logical_and(data['Family']<1.0 ,data['Generosity']>0.3 ))