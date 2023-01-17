# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename));

        

Data_vg = pd.read_csv("../input/videogamesales/vgsales.csv");

Data_vg
Data_vg.dropna(inplace=True)

Data_vg.drop(columns="Rank",inplace=True)

Data_vg = Data_vg[Data_vg["Year"]<2017.0]

Data_vg
matrix = Data_vg.corr()

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(matrix,annot=True,cmap="YlGn_r")
# Summary

Data_vg.describe()
sns.pairplot(Data_vg);
df = Data_vg.groupby(by  = 'Year').sum()

df.plot.line(figsize=(10,10), grid="on");

plt.ylabel("Sales in million $");
df = pd.DataFrame(Data_vg['Genre'].value_counts(normalize=True))

plot = df.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(10, 10))
df1 = pd.DataFrame(Data_vg.groupby('Platform')['Genre'].nunique())

df1.sort_values(by=['Genre'], inplace=True)

df1[df1["Genre"]>11]
df = pd.DataFrame(Data_vg["Name"].value_counts().head(5))

df
title = {'family': 'serif',

        'color':  'darkblue',

        'weight': 'normal',

        'size': 16,

        }

sub_head = {'family': 'monospace',

        'color':  'darkblue',

        'size': 16,

        'weight': 'demibold',

        }
dt = pd.DataFrame(Data_vg['Year'].value_counts()).sort_index().tail(15)

dt = list(dt["Year"])
plt.figure(figsize=(16,8))

sns.countplot("Year",data=Data_vg)

plt.xlim([21.5,31.5])

plt.xlabel("no. of games launched")

li=21.9

for i in range(10):

    plt.text(li, dt[i], dt[i])

    li+=1

plt.title("10 Most frequent launch years",fontdict=title)

plt.ylabel("Year of release");
dft = Data_vg[Data_vg["Year"]!=1981.0]

dt = pd.DataFrame(dft['Year'].value_counts()).sort_index().head(10)

dt = list(dt["Year"])
plt.figure(figsize=(16,8))

sns.countplot("Year",data=dft)

plt.xlim([-0.5,9.5])

plt.ylim([0,50])

li=-0.1

for i in range(10):

    plt.text(li, dt[i], '['+ str(dt[i]) + ']' )

    li+=1

plt.xlabel("no. of games launched")

plt.title("10 Least frequent launch years",fontdict=sub_head)

plt.ylabel("Year of release");
Data_vg[Data_vg["Global_Sales"]<1.5].hist(column="Global_Sales",bins = 20, color= 'orange' )

plt.xlabel("Sales in million less than 1.5 million $")

plt.title("Worldwide Sales",fontdict=title);
df1 = pd.DataFrame(Data_vg.groupby('Publisher')['Global_Sales'].sum())

df1.sort_values(by=['Global_Sales'], inplace=True)

df1 = df1.tail(10)

plot = df1.plot.pie(y='Global_Sales', autopct='%1.1f%%', figsize=(10, 10))

plt.title("Publisher market share",fontdict=title);
df = Data_vg.drop(columns=["Year","Global_Sales"]).head(10)

ax = df.plot.bar(x="Name",stacked=True,rot=85)

plt.title("Top 10 Games globally",fontdict=title);
df1 = pd.DataFrame(Data_vg.groupby('Name')['NA_Sales'].sum())

df1.sort_values(by=['NA_Sales'], inplace=True)

df1 = df1.tail(5)

df1.plot.pie(y='NA_Sales', autopct='%1.1f%%', figsize=(6, 6))

plt.title("Best selling games in North America", fontdict=title)



df1 = pd.DataFrame(Data_vg.groupby('Name')['EU_Sales'].sum())

df1.sort_values(by=['EU_Sales'], inplace=True)

df1 = df1.tail(5)

df1.plot.pie(y='EU_Sales', autopct='%1.1f%%', figsize=(6, 6))

plt.title("Best selling games in Europe", fontdict=title)



df1 = pd.DataFrame(Data_vg.groupby('Name')['JP_Sales'].sum())

df1.sort_values(by=['JP_Sales'], inplace=True)

df1 = df1.tail(5)

df1.plot.pie(y='JP_Sales', autopct='%1.1f%%', figsize=(6, 6))

plt.title("Best selling games in Japan", fontdict=title);