import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
df_athletes = pd.read_csv("../input/athlete_events.csv")
df_athletes.head()
#How many events do we have thus far, in Olympics
print("There are {} unique sports thus far, in Olympics". format(df_athletes["Sport"].nunique()))
#Not all of the above sports were part of Olympics starting 1896. Let us find out which year the sport was inducted into Olympics
df_athletes.groupby("Sport")["Year"].min().sort_values().reset_index()
df_sports = df_athletes[df_athletes["Year"]<1950].groupby("Sport")["Year"].min().sort_values(ascending=False).reset_index()
plt.figure(figsize=(15,20))
plt.xlim(1700,1950)
sns.barplot(x = "Year", y = "Sport", hue="Year", data=df_sports)  
plt.show()
df_sports = df_athletes[df_athletes["Year"]>=1950].groupby("Sport")["Year"].min().sort_values(ascending=False).reset_index()
plt.figure(figsize=(15,20))
plt.xlim(1930,2017)
sns.barplot(x = "Year", y = "Sport", hue="Year", data=df_sports)
plt.show()
#Let us summarize number of "Sports" held per year, starting 1896
df_sports = df_athletes.groupby("Year")["Sport"].nunique().sort_values(ascending=False).reset_index()
fig = plt.figure(figsize=(30,10))
fig.add_subplot(111)
sns.barplot(x = "Year", y = "Sport", data=df_sports, palette="summer", saturation=.65 )
#Now that we have looked the events across the years, let us look at the gender participation in these events
df_sports = df_athletes.groupby(["Year","Sex"])["Sport"].count().reset_index().rename(columns={"Sport":"Count"})
plt.figure(figsize=(25,10))
sns.barplot(x="Year", y = "Count", hue = "Sex", data = df_sports, palette = "Blues" )
#Let us look at the minimum age of participant, per Gender, in Olympics
df_sports = df_athletes.groupby(["Year","Sex"])["Age"].min().reset_index()
plt.figure(figsize=(40,10))
plt.subplot(121)
sns.barplot(x="Year", y = "Age", hue = "Sex", data = df_sports, palette = "Blues" )
#1. In year 1986, we had a 'Male' participant around 10 years old. 
df_athletes[df_athletes["Age"]<11]  [["Year","Name","Sex","Age","Games","Season","City","Sport","Event","Medal"]]
#1. We have few female participants around 11 years old. The names repeat because they participated in more than one event.
df_athletes[(df_athletes["Sex"]=="F") & (df_athletes["Age"]<12)  &  (df_athletes["Year"].isin ([1924, 1928, 1932, 1936, 1960, 1968])  )]  [["Year","Name","Sex","Age","Games","Season","City","Sport","Event","Medal"]].sort_values(by="Name")
#Let us look at the minimum age of participant, per Gender, in Olympics
df_sports = df_athletes.groupby(["Year","Sex"])["Age"].max().reset_index()
plt.figure(figsize=(40,10))
plt.subplot(221)
sns.barplot(x="Year", y = "Age", hue = "Sex", data = df_sports, palette = "Blues" )
#1. In few Olympics, we had 'Male' participants who are quite old
df_athletes[df_athletes["Age"]>80]  [["Year","Name","Sex","Age","Games","Season","City","Sport","Event","Medal"]].sort_values(by="Name")
#1. In few Olympics, we had 'Female' participants who are quite old
df_athletes[(df_athletes["Sex"]=="F") & (df_athletes["Age"]>70) ]  [["Year","Name","Sex","Age","Games","Season","City","Sport","Event","Medal"]].sort_values(by="Name")
#Let us look at the minimum height of the participants
df_sports = df_athletes.groupby(["Year","Sex"])["Height"].min().reset_index()
plt.figure(figsize=(40,10))
plt.subplot(121)
sns.barplot(x="Year", y = "Height", hue = "Sex", data = df_sports, palette = "Blues" )
#Let us look at the maximum height of the participants
df_sports = df_athletes.groupby(["Year","Sex"])["Height"].max().reset_index()
plt.figure(figsize=(40,10))
plt.subplot(121)
sns.barplot(x="Year", y = "Height", hue = "Sex", data = df_sports, palette = "Blues" )
#1. Tallest male heights are seen in "Basketball"
df_athletes[(df_athletes["Sex"]=="M") & (df_athletes["Height"]>220) ]  [["Year","Name","Sex","Age","Height","Games","Season","City","Sport","Event","Medal"]].sort_values(by="Year")
#1. Tallest female height is seen in "Basketball"
df_athletes[(df_athletes["Sex"]=="F") & (df_athletes["Height"]>210) ]  [["Year","Name","Sex","Age","Height","Games","Season","City","Sport","Event","Medal"]].sort_values(by="Year")
#How many Cities are there since 1896?
print("There are {} cities in whicn Olympics was conducted since 1896".format (df_athletes["City"].value_counts().nunique()))

#How many cities in which Olympics was conducted more than once?
df_athletes.groupby(["City"])["Year"].nunique().sort_values(ascending=False)
#Which team won the most Gold medals, in each of the Olympics?
df_sports  = df_athletes[df_athletes["Medal"]=="Gold"].groupby(["Year","Team"], as_index=False)["ID"].count().rename(columns={"ID":"Total"})
df_sports.iloc[df_sports.groupby(["Year"])["Total"].idxmax()]