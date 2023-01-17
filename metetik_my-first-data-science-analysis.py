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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df15 = pd.read_csv('../input/2015.csv')
df16 = pd.read_csv('../input/2016.csv')
df17 = pd.read_csv('../input/2017.csv')
df15.info()
print() #for separate lines
print() 

df16.info()
print()
print()

df17.info()
display(df15.head())

display(df16.head())

display(df17.head())
print(df15.Region.unique())

print(df16.Region.unique())
df15.rename(columns={"Happiness Score" : "HScore"}, inplace=True)

df16.rename(columns={"Happiness Score" : "HScore"}, inplace=True)

df17.rename(columns={"Happiness.Score" : "HScore"}, inplace=True)

WEurope15 = df15[df15.Region == "Western Europe"]

NAmerica15 = df15[df15.Region == "North America"]

Australia15 = df15[df15.Region == "Australia and New Zealand"]

MEast15 = df15[df15.Region == "Middle East and Northern Africa"]

LAmerica15 = df15[df15.Region == "Latin America and Caribbean"]

SEAsia15 = df15[df15.Region == "Southeastern Asia"]

CEurope15 = df15[df15.Region == "Central and Eastern Europe"]

EAsia15 = df15[df15.Region == "Eastern Asia"]

SAfrica15 = df15[df15.Region == "Sub-Saharan Africa"]

SAsia15 = df15[df15.Region == "Southern Asia"]

WEurope16 = df16[df16.Region == "Western Europe"]

NAmerica16 = df16[df16.Region == "North America"]

Australia16 = df16[df16.Region == "Australia and New Zealand"]

MEast16 = df16[df16.Region == "Middle East and Northern Africa"]

LAmerica16 = df16[df16.Region == "Latin America and Caribbean"]

SEAsia16 = df16[df16.Region == "Southeastern Asia"]

CEurope16 = df16[df16.Region == "Central and Eastern Europe"]

EAsia16 = df16[df16.Region == "Eastern Asia"]

SAfrica16 = df16[df16.Region == "Sub-Saharan Africa"]

SAsia16 = df16[df16.Region == "Southern Asia"]
MeanRegions15 = np.array([WEurope15.HScore.mean(),NAmerica15.HScore.mean(),Australia15.HScore.mean(),MEast15.HScore.mean(),LAmerica15.HScore.mean(),SEAsia15.HScore.mean(),CEurope15.HScore.mean(),EAsia15.HScore.mean(),SAfrica15.HScore.mean(),SAsia15.HScore.mean()])

Regions15 = ["Western Europe","North America","Australia and New Zealand","Middle East and Northern Africa","Latin America and Caribbean","Southeastern Asia","Central and Eastern Europe","Eastern Asia","Sub-Saharan Africa","Southern Asia"]

plt.figure(figsize=(30,15))
plt.bar(Regions15,MeanRegions15)
plt.title("Happiness Score of Regions in 2015")
plt.xlabel("Regions")
plt.ylabel("Mean of Happiness Score")
plt.show()

MeanRegions16 = np.array([WEurope16.HScore.mean(),NAmerica16.HScore.mean(),Australia16.HScore.mean(),MEast16.HScore.mean(),LAmerica16.HScore.mean(),SEAsia16.HScore.mean(),CEurope16.HScore.mean(),EAsia16.HScore.mean(),SAfrica16.HScore.mean(),SAsia16.HScore.mean()])

Regions16 = ["Western Europe","North America","Australia and New Zealand","Middle East and Northern Africa","Latin America and Caribbean","Southeastern Asia","Central and Eastern Europe","Eastern Asia","Sub-Saharan Africa","Southern Asia"]

plt.figure(figsize=(30,15))
plt.bar(Regions16,MeanRegions16)
plt.title("Happiness Score of Regions in 2016")
plt.xlabel("Regions")
plt.ylabel("Mean of Happiness Score")
plt.show()

display(df15.corr())

corrmap = df15.corr()

plt.subplots(figsize=(18, 18))

sns.heatmap(corrmap, vmax=.9,annot=True,linewidths=.5)
display(df16.corr())

corrmap = df16.corr()

plt.subplots(figsize=(18, 18))

sns.heatmap(corrmap, vmax=.9,annot=True,linewidths=.5)
display(df17.corr())

corrmap = df17.corr()

plt.subplots(figsize=(18, 18))
             
sns.heatmap(corrmap, vmax=.9,annot=True,linewidths=.5)
df15.rename(columns={"Economy (GDP per Capita)" : "Economy"}, inplace=True)

df16.rename(columns={"Economy (GDP per Capita)" : "Economy"}, inplace=True)

df17.rename(columns={"Economy..GDP.per.Capita." : "Economy"}, inplace=True)

df15.plot(kind='scatter', x='Economy', y='HScore',alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,1))
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title('2015')

df16.plot(kind='scatter', x='Economy', y='HScore',alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,2))
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title('2016')

df17.plot(kind='scatter', x='Economy', y='HScore',alpha = 0.5,color = 'red',figsize=(12,9),subplots = (3,1,3))
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title('2017')
plt.figure(figsize=(20,20))
plt.subplot(3,1,1)
plt.plot(df15.HScore,df15.Family,color="red",label= "2015")
plt.xlabel("Happiness Score")
plt.ylabel("Family")
plt.subplot(3,1,2)
plt.plot(df16.HScore,df16.Family,color="green",label= "2016")
plt.xlabel("Happiness Score")
plt.ylabel("Family")
plt.subplot(3,1,3)
plt.plot(df17.HScore,df17.Family,color="blue",label= "2017")
plt.xlabel("Happiness Score")
plt.ylabel("Family")
plt.show()
