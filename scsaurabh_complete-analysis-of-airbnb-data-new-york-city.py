import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.head()
df.isnull().sum()
df.describe()
df.info()
df['reviews_per_month'].fillna(0,inplace = True)
df['name'].fillna("$",inplace=True)

df['host_name'].fillna("#",inplace=True)
df.drop(['last_review'],axis=1,inplace=True)
df.head()
df.neighbourhood.unique()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(df.neighbourhood_group,palette="muted")

plt.show()
df1 = df[df.neighbourhood_group == "Brooklyn"][["neighbourhood","price"]]

d = df1.groupby("neighbourhood").mean()

sns.distplot(d)

plt.show()
f,ax = plt.subplots(figsize=(15,4))

df1 = df[df.neighbourhood_group=="Brooklyn"]['price']

sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(15,4))

df1 = df[df.neighbourhood_group=="Manhattan"]['price']

sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(15,4))

df1 = df[df.neighbourhood_group=="Queens"]['price']

sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(15,4))

df1 = df[df.neighbourhood_group=="Staten Island"]['price']

sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(15,4))

df1 = df[df.neighbourhood_group=="Bronx"]['price']

sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(12,5))

ax = sns.countplot(df.room_type,palette="muted")

plt.show()
df1 = df[df.room_type == "Private room"][["neighbourhood_group","price"]]

d = df1.groupby("neighbourhood_group").mean()

sns.distplot(d)

plt.show()
df1 = df[df.room_type=='Private room']['price']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.distplot(df1)

plt.show()
df1 = df[df.room_type=='Shared room']['price']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.distplot(df1)

plt.show()
df1 = df[df.room_type=='Entire home/apt']['price']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.distplot(df1)

plt.show()
f,ax = plt.subplots(figsize=(15,5))

ax = sns.distplot(df.reviews_per_month)

plt.show()
f,ax = plt.subplots(figsize=(15,5))

ax = sns.distplot(df.availability_365)

plt.show()
df1 = df[df.room_type=="Private room"]['minimum_nights']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.swarmplot(y= df1.index,x= df1.values)

plt.xlabel("minimum_nights")

plt.show()
df1 = df[df.room_type=="Shared room"]['minimum_nights']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.swarmplot(y= df1.index,x= df1.values)

plt.xlabel("minimum_nights")

plt.show()
df1 = df[df.room_type=="Entire home/apt"]['minimum_nights']

f,ax = plt.subplots(figsize=(15,5))

ax = sns.swarmplot(y= df1.index,x= df1.values)

plt.xlabel("minimum_nights")

plt.show()
f,ax = plt.subplots(figsize=(16,8))

ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.neighbourhood_group,palette="coolwarm")

plt.show()
f,ax = plt.subplots(figsize=(16,8))

ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.availability_365,palette="coolwarm")

plt.show()
df1 = df.host_id.value_counts()[:11]

f,ax = plt.subplots(figsize=(16,5))

ax = sns.barplot(x = df1.index,y=df1.values,palette="muted")

plt.show()