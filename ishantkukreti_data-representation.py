import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
df=pd.read_csv("../input/bike-sharing/day.csv")

df.shape

df.head()

df.describe()
for i in df.columns.values:

    print(i)

    print(df[i].isnull().value_counts(),end='\n\n')
df=df.drop(["instant",'dteday','casual','registered'],axis=1)
df.head()
for i in df.columns.values:

    print(i)

    print(df[i].value_counts(),end="\n\n")


df["season"].replace(to_replace=(1,2,3,4),value=("winter","summer","autumn","spring"),inplace=True)

df["yr"].replace(to_replace=(0,1),value=(2018,2019),inplace=True)

df["mnth"].replace(to_replace=(1,2,3,4,5,6,7,8,9,10,11,12),value=("jan","feb","march","april","may","june","july","aug","sept","oct","nov","dec"),inplace=True)

df["weekday"].replace(to_replace=(0,1,2,3,4,5,6),value=("sun","mon","tue","wed","thu","fri","sat"),inplace=True)

df["weathersit"].replace(to_replace=(1,2,3),value=("clear","mist","snow"),inplace=True)

df_cat=df[["season","yr","mnth","holiday","weekday","weathersit"]]#catagorical features 

df_num=df[["atemp","hum","windspeed"]]#numeric features 

y=df["cnt"]#label
for i in df_cat.columns.values:

    print(i)

    print(df[i].value_counts(),end="\n\n")
for i in df_cat.columns.values:

    sns.boxplot(x=df_cat[i],y=y)

    plt.show()

    
for i in df_num.columns.values:

    sns.regplot(x=df_num[i],y=y)

    plt.show()
df = pd.get_dummies(df, drop_first=True)

df.info()
df.head()
plt.figure(figsize=(25,25))

sns.heatmap(df.corr(),annot=True,cmap="BuGn")

plt.show()