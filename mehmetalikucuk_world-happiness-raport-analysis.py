# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data15 = pd.read_csv("../input/2015.csv")
data16 = pd.read_csv("../input/2016.csv")
data17 = pd.read_csv("../input/2017.csv")
print(data15.columns)
print(data16.columns)
print(data17.columns)
data15.rename(columns={"Happiness Rank" : "HappRank",
                     "Happiness Score" : "HappScr",
                     "Economy (GDP per Capita)" : "Economy",
                     "Health (Life Expectancy)" : "Health",
                     "Trust (Government Corruption)" : "Trust",
                     "Dystopia Residual" : "DystResd"}, inplace=True)
data15.drop(columns="Standard Error", inplace=True)
data15.head()
data16.rename(columns={"Happiness Rank" : "HappRank",
                     "Happiness Score" : "HappScr",
                     "Standard Error" : "StdErr",
                     "Economy (GDP per Capita)" : "Economy",
                     "Health (Life Expectancy)" : "Health",
                     "Trust (Government Corruption)" : "Trust",
                     "Dystopia Residual" : "DystResd"}, inplace=True)
data16.drop(columns=["Lower Confidence Interval","Upper Confidence Interval"], inplace=True)
data16.head()
data17.rename(columns={"Happiness.Rank" : "HappRank",
                     "Happiness.Score" : "HappScr",
                     "Economy..GDP.per.Capita." : "Economy",
                     "Health..Life.Expectancy." : "Health",
                     "Trust..Government.Corruption." : "Trust",
                     "Dystopia.Residual" : "DystResd"}, inplace=True)
data17.drop(columns=["Whisker.high","Whisker.low"], inplace=True)
data17.head()
# Correlation in dataset 2015
data15.corr()
print("in 2015 dataset happiness max corr: ", data15.corr().HappScr[2:].idxmax()
     ,", Value: ", data15.corr().HappScr[2:].max())
print("in 2016 dataset happiness max corr: ", data16.corr().HappScr[2:].idxmax()
     ,", Value: ", data16.corr().HappScr[2:].max())
print("in 2017 dataset happiness max corr: ", data17.corr().HappScr[2:].idxmax()
     ,", Value: ", data17.corr().HappScr[2:].max())
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.scatter(x=data15.Economy, y=data15.HappScr, alpha=0.6, color="red")
plt.xlabel("Economy (2015)")
plt.ylabel("Happiness Score")
plt.subplot(132)
plt.scatter(x=data16.Economy, y=data16.HappScr, alpha=0.6, color="red")
plt.xlabel("Economy (2016)")
plt.title("Relationship between Happiness Score and Economy")
plt.subplot(133)
plt.scatter(x=data17.Economy, y=data17.HappScr, alpha=0.6, color="red")
plt.xlabel("Economy (2017)")
plt.show()
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.plot(data15.HappRank, data15.Economy, color="blue")
plt.axis(ymin=0, ymax=2)
plt.xlabel("Happiness Rank (2015)")
plt.ylabel("Economy (GDP per Capita)")
plt.grid(True)
plt.subplot(132)
plt.plot(data16.HappRank, data16.Economy, color="blue")
plt.axis(ymin=0, ymax=2)
plt.xlabel("Happiness Rank (2016)")
plt.title("dd")
plt.grid(True)
plt.subplot(133)
plt.plot(data17.HappRank, data17.Economy, color="blue")
plt.axis(ymin=0, ymax=2)
plt.xlabel("Happiness Rank (2017)")
plt.grid(True)
plt.show()

plt.figure(figsize=(20,10))
plt.hist(x=data15.Region[:50], bins = 30)    # First 50 countries
plt.title("Distribution of The First Fifty Happy Countries by Region (2015)")
plt.xlabel("Regions")
plt.ylabel("Countries")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20,10))
plt.hist(x=data16.Region[:50], bins = 30)    # First 50 countries
plt.title("Distribution of The First Fifty Happy Countries by Region (2016)")
plt.xlabel("Regions")
plt.ylabel("Countries")
plt.xticks(rotation=45)
plt.show()