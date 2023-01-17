# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import folium

from folium import plugins



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/house-prices-turkey/ads.csv")

df.head()
df=df.rename(columns={"developer_id":"developer","city_id":"city","district_id":"district","Name":"name"})

df.head()
location=df[["latitude","longitude"]]
m = folium.Map(location=[40, 32],zoom_start=7)

for i in range(0,len(df)):

    folium.Marker([df.iloc[i]['latitude'], df.iloc[i]['longitude']], popup=df.iloc[i]['name']).add_to(m)

m
m = folium.Map(location=[40, 32],zoom_start=7)

for i in range(0,len(df)):

    folium.CircleMarker(location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],radius=15, fill_color='red',fill=True).add_to(m)

m
m = folium.Map(location=[40, 32],zoom_start=7)

for i in range(0,len(df)):

    folium.Marker([df.iloc[i]['latitude'], df.iloc[i]['longitude']], popup=df.iloc[i]['name']).add_to(m)

    folium.CircleMarker(location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],radius=30, fill_color='red',fill=True).add_to(m)

m
m = folium.Map([40 ,32], zoom_start=7,width="%100",height="%100")



plugins.MarkerCluster(location).add_to(m)



m
m=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)

heat_data=list(zip(location))

folium.plugins.HeatMap(location).add_to(m)

m
df.head()
df.drop(["id","ad_id","currency","developer","latitude","longitude"],axis=1,inplace=True)
fig,ax=plt.subplots(ncols=2,figsize=(16,5))

sns.countplot(df["city"],ax=ax[1],order = df['city'].value_counts().index)

df["city"].value_counts().plot.pie(ax=ax[0],explode=(0,0.1,0.2,0.5));

plt.figure(figsize=(16,5))

sns.barplot(x=df["city"],y=df["total_land"],palette="Set2");
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.barplot(x=df["city"],y=df["max_unit_price"],palette="Set2",ax=ax[0])

sns.barplot(x=df["city"],y=df["min_unit_price"],palette="Set2",ax=ax[1]);
plt.figure(figsize=(16,5))

sns.countplot(df[df["city"]=="İstanbul"]["district"])

plt.xticks(rotation=75);
plt.figure(figsize=(16,5))

sns.swarmplot(data=df[df["city"]=="İstanbul"],x="district",y="total_land")

plt.xticks(rotation=75);
plt.figure(figsize=(16,5))

sns.pointplot(data=df[df["city"]=="İstanbul"],x="district",y="total_land",ci=None)

plt.xticks(rotation=75);
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,5))

sns.distplot(df["max_unit_price"],kde=False,ax=ax[0],color="#7D3C98",bins=10)

sns.distplot(df["min_unit_price"],kde=False,ax=ax[1],color="#229954",bins=10);
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,5))

sns.kdeplot(df["max_unit_price"],shade=True,ax=ax[0],color="#7D3C98")

sns.kdeplot(df["min_unit_price"],shade=True,ax=ax[1],color="#229954");
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.stripplot(x="city",y="max_unit_price",data=df,ax=ax[0])

sns.stripplot(x="city",y="min_unit_price",data=df,ax=ax[1]);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.lineplot(x="city",y="max_unit_price",data=df,ax=ax[0],color="#7D3C98",lw=5)

sns.lineplot(x="city",y="min_unit_price",data=df,ax=ax[0],color="#229954",lw=5)

sns.pointplot(x="city",y="max_unit_price",data=df,ax=ax[1],color="#7D3C98",lw=5)

sns.pointplot(x="city",y="min_unit_price",data=df,ax=ax[1],color="#229954",lw=5);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.pointplot(x="city",y="max_unit_price",data=df,ax=ax[0],color="#7D3C98")

sns.pointplot(x="city",y="min_unit_price",data=df,ax=ax[1],color="#229954");
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.boxplot(x="city",y="max_unit_price",data=df,ax=ax[0],color="#7D3C98")

sns.boxplot(x="city",y="min_unit_price",data=df,ax=ax[1],color="#229954");
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.violinplot(x="city",y="max_unit_price",data=df,ax=ax[0],color="#7D3C98")

sns.violinplot(x="city",y="min_unit_price",data=df,ax=ax[1],color="#229954");
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,5))

sns.scatterplot(x="min_unit_price",y="max_unit_price",data=df,ax=ax[0])

sns.scatterplot(x="min_unit_price",y="max_unit_price",data=df,ax=ax[1],hue="city");
num_data=list(df.select_dtypes(["int64","float64"]).columns)

num_data.remove("total_units")
sns.pairplot(df,vars=df[num_data]);
sns.pairplot(df,vars=df[num_data],hue="city");