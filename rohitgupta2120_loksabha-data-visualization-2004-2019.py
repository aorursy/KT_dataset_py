import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_2004 = pd.read_csv("../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv")

df_2009 = pd.read_csv("../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2009.csv")

df_2014 = pd.read_csv("../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2014.csv")

df_2019 = pd.read_csv("../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2019.csv")
df_2004["Year"] = 2004

df_2009["Year"] = 2009

df_2014["Year"] = 2014

df_2019["Year"] = 2019

Frames = [df_2004,df_2009,df_2014,df_2019]
Data = pd.concat(Frames)

Data
df1 = pd.DataFrame(Data['Education'].value_counts(normalize=True),

                   index=['Doctorate','Graduate Professional', 'Graduate','Post Graduate','12th Pass','10th Pass','8th Pass','5th Pass','Literate','Not Given','Illiterate','Others'])

plot = df1.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(10, 10))
sns.pairplot(Data);
Data['Education'].value_counts().plot(kind='barh', figsize=(20,10))

plt.xlabel("no. of candidates")

plt.ylabel("qualificaton");
Data.hist(bins = 30, figsize=(15,15), color= 'Blue');
df1 = pd.DataFrame(Data.groupby('Party')['Criminal Cases'].nunique())

df1.sort_values(by=['Criminal Cases'], inplace=True)

df1
df1[df1['Criminal Cases']>=10]
df1[df1['Criminal Cases']==1]
df2 = pd.DataFrame(Data.groupby('City')['Criminal Cases'].nunique())

df2.sort_values(by=['Criminal Cases'], inplace=True)

df2 = df2.tail(10)

df2["City"]=df2.index
dt=df2["Criminal Cases"]

plt.figure(figsize=(12,8))

sns.barplot(y="City",x="Criminal Cases",data=df2)

plt.xlim([5,10])

plt.xlabel("no. of games launched")

li=0.0

for i in range(10):

    plt.text(dt[i],li, dt[i])

    li+=1

plt.title("10 Most frequent launch years")

plt.ylabel("Year of release");
df3 = Data.sort_values(by=['Criminal Cases']).tail(10)

df3 
plt.figure(figsize=(12,8))

sns.catplot(x="Criminal Cases",y="Candidate",data=df3, kind="bar");

plt.ylabel("Candidates");