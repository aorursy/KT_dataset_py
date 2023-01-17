import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os 

import seaborn as sns

import geopandas as gpd

import folium

from folium import plugins

import datetime

import math
os.listdir("../input/chicago-food-inspections")
data=pd.read_csv("../input/chicago-food-inspections/food-inspections.csv")

data.head()
len(data)
plt.figure(figsize=(10,10))

sns.barplot(x=data.isnull().sum().sort_values(ascending=False),y=data.isnull().sum().sort_values(ascending=False).index)

plt.title("counts of missing value",size=20)

data=data.dropna(subset=['Violations','Facility Type','Latitude','Longitude','AKA Name'])

data.isnull().sum()
data['year']=data['Inspection Date'].apply(lambda x:x.split('-')[0])

data['month']=data['Inspection Date'].apply(lambda x:x.split('-')[1])

data['day']=data['Inspection Date'].apply(lambda x:x.split('-')[2].split('T')[0])

data.head()
fig,ax=plt.subplots(2,2,figsize=(20,20))

x=data.year.value_counts().index

y=data.year.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,0])

ax[0,0].set_title("The counts of inspection by year",size=20)

ax[0,0].set_ylabel('counts',size=18)

ax[0,0].set_xlabel('')



x=data.month.value_counts().index

y=data.month.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,1])

ax[0,1].set_title("The counts of inspection by month",size=20)

ax[0,1].set_ylabel('counts',size=18)

ax[0,1].set_xlabel('')



x=data.day.value_counts().index

y=data.day.value_counts()

sns.barplot(x=x,y=y,ax=ax[1,0])

ax[1,0].set_title("The counts of inspection by day",size=20)

ax[1,0].set_ylabel('counts',size=18)

ax[1,0].set_xlabel('')



data.groupby(['year','month'])['Inspection ID'].agg('count').unstack('year').plot(ax=ax[1,1])

ax[1,1].set_title("The counts of inspection for every month by year",size=20)

ax[1,1].set_ylabel('counts',size=18)

ax[1,1].set_xlabel('month')
fig,ax=plt.subplots(2,2,figsize=(15,16))

data.Risk.value_counts().plot(kind='bar',color=['red','yellow','green'],ax=ax[0,0])

ax[0,0].tick_params(axis='x',labelrotation=360)

ax[0,0].set_title("The counts of Risk",size=20)

ax[0,0].set_ylabel('counts',size=18)





data.groupby(['year','Risk'])['Inspection ID'].agg('count').unstack('Risk').plot(ax=ax[0,1],color=['red','yellow','green'])

ax[0,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.15,0.75))

ax[0,1].set_title("The counts of Risk by year",size=20)

ax[0,1].set_ylabel('counts',size=18)



data.groupby(['month','Risk'])['Inspection ID'].agg('count').unstack('Risk').plot(ax=ax[1,0],color=['red','yellow','green'])

ax[1,0].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(-0.25,0.75))

ax[1,0].set_title("The counts of Risk by month",size=20)

ax[1,0].set_ylabel('counts',size=18)



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk' ,data=data, ax=ax[1,1])

ax[1,1].set_title("The distribution of inspections by risk",size=20)

ax[1,1].set_xlabel('Longitude')

ax[1,1].set_ylabel('LATITUDE')

data_risk1=data[data.Risk=='Risk 1 (High)']

data_risk1.head()
fig,ax=plt.subplots(1,2,figsize=(15,8))

sns.barplot(x=data_risk1['Facility Type'].value_counts()[:10],y=data_risk1['Facility Type'].value_counts()[:10].index,ax=ax[0])

ax[0].set_title("Top 10 Facility Type by the counts of risk 1 ",size=20)

ax[0].set_xlabel('counts',size=18)





count=data_risk1.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False)

groups=list(data_risk1.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False).index[:10])

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

ax[1].set_ylabel('')

ax[1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.15,1.2))
data_risk1_2000=data_risk1[:2000]

Long=data_risk1_2000.Longitude.mean()

Lat=data_risk1_2000.Latitude.mean()

risk1_map=folium.Map([Lat,Long],zoom_start=12)



risk1_distribution_map=plugins.MarkerCluster().add_to(risk1_map)

for lat,lon,label in zip(data_risk1_2000.Latitude,data_risk1_2000.Longitude,data_risk1_2000['AKA Name']):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(risk1_distribution_map)

risk1_map.add_child(risk1_distribution_map)



risk1_map
data_risk2=data[data.Risk=='Risk 2 (Medium)']

data_risk2.head()
fig,ax=plt.subplots(1,2,figsize=(15,8))

sns.barplot(x=data_risk2['Facility Type'].value_counts()[:10],y=data_risk2['Facility Type'].value_counts()[:10].index,ax=ax[0])

ax[0].set_title("Top 10 Facility Type by the counts of risk 2 ",size=20)

ax[0].set_xlabel('counts',size=18)





count=data_risk2.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False)

groups=list(data_risk2.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False).index[:10])

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

ax[1].set_ylabel('')

ax[1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.15,1.2))
data_risk2_2000=data_risk2[:2000]

Long=data_risk2_2000.Longitude.mean()

Lat=data_risk2_2000.Latitude.mean()

risk2_map=folium.Map([Lat,Long],zoom_start=12)



risk2_distribution_map=plugins.MarkerCluster().add_to(risk2_map)

for lat,lon,label in zip(data_risk2_2000.Latitude,data_risk2_2000.Longitude,data_risk2_2000['AKA Name']):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(risk2_distribution_map)

risk2_map.add_child(risk2_distribution_map)



risk2_map
data_risk3=data[data.Risk=='Risk 3 (Low)']

data_risk3.head()
fig,ax=plt.subplots(1,2,figsize=(15,8))

sns.barplot(x=data_risk3['Facility Type'].value_counts()[:10],y=data_risk3['Facility Type'].value_counts()[:10].index,ax=ax[0])

ax[0].set_title("Top 10 Facility Type by the counts of risk 3 ",size=20)

ax[0].set_xlabel('counts',size=18)





count=data_risk3.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False)

groups=list(data_risk3.groupby(['Facility Type'])['Inspection ID'].agg('count').sort_values(ascending=False).index[:10])

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

ax[1].set_ylabel('')

ax[1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.15,1.2))
data_risk3_2000=data_risk3[:2000]

data_risk3_2000['AKA Name']=data_risk3_2000['AKA Name'].apply(lambda x:x.strip('`').strip())

Long=data_risk3_2000.Longitude.mean()

Lat=data_risk3_2000.Latitude.mean()

risk3_map=folium.Map([Lat,Long],zoom_start=12)



risk3_distribution_map=plugins.MarkerCluster().add_to(risk3_map)

for lat,lon,label in zip(data_risk3_2000.Latitude,data_risk3_2000.Longitude,data_risk3_2000['AKA Name']):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(risk3_distribution_map)

risk3_map.add_child(risk3_distribution_map)



risk3_map
fig,ax=plt.subplots(2,2,figsize=(20,16))

y=data['Facility Type'].value_counts()[:10].index

x=data['Facility Type'].value_counts()[:10]

sns.barplot(x=x,y=y,ax=ax[0,0])

ax[0,0].set_title("Top 10 Facility Type by the counts of inspection ",size=20)

ax[0,0].set_xlabel('counts',size=18)

ax[0,0].set_ylabel('')



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk',hue_order=['Risk 1 (High)','Risk 2 (Medium)','Risk 3 (Low)'] ,data=data[data['Facility Type']=='Restaurant'], ax=ax[0,1])

ax[0,1].set_title("The distribution of inspections for restaurant",size=20)

ax[0,1].set_xlabel('Longitude')

ax[0,1].set_ylabel('LATITUDE')



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk' ,hue_order=['Risk 1 (High)','Risk 2 (Medium)','Risk 3 (Low)'],data=data[data['Facility Type']=='Grocery Store'], ax=ax[1,0])

ax[1,0].set_title("The distribution of inspections for Grocery Store",size=20)

ax[1,0].set_xlabel('Longitude')

ax[1,0].set_ylabel('LATITUDE')



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk',hue_order=['Risk 1 (High)','Risk 2 (Medium)','Risk 3 (Low)'] ,data=data[data['Facility Type']=='School'], ax=ax[1,1])

ax[1,1].set_title("The distribution of inspections for School",size=20)

ax[1,1].set_xlabel('Longitude')

ax[1,1].set_ylabel('LATITUDE')
fig,ax=plt.subplots(2,2,figsize=(20,16))

x=data.Results.value_counts().index

y=data.Results.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,0])

ax[0,0].set_title("The counts of Results of inspection ",size=20)

ax[0,0].set_ylabel('counts',size=18)

ax[0,0].set_xlabel('')



data.groupby(['Results','year'])['Inspection ID'].agg('count').unstack('Results').plot(kind='bar',ax=ax[0,1])

ax[0,1].tick_params(axis='x',labelrotation=360)

ax[0,1].legend(loc=0, ncol=1, fontsize=14,bbox_to_anchor=(1.15,0.75))

ax[0,1].set_title("The counts of results of inspection by year ",size=20)

ax[0,1].set_ylabel('counts',size=18)



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk' ,hue_order=['Risk 1 (High)','Risk 2 (Medium)','Risk 3 (Low)'],data=data[data.Results=='Pass'], ax=ax[1,0])

ax[1,0].set_title("The distribution of result is pass",size=20)

ax[1,0].set_xlabel('Longitude')

ax[1,0].set_ylabel('LATITUDE')



sns.scatterplot(x='Longitude',y='Latitude',hue='Risk',hue_order=['Risk 1 (High)','Risk 2 (Medium)','Risk 3 (Low)'] ,data=data[data.Results=='Fail'], ax=ax[1,1])

ax[1,1].set_title("The distribution of result is fail",size=20)

ax[1,1].set_xlabel('Longitude')

ax[1,1].set_ylabel('LATITUDE')