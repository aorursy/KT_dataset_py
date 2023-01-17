import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import os

import seaborn as sns

import geopandas

import folium

sns.set(style="darkgrid")
data = pd.read_csv("../input/as_seekers_monthly.csv")

print(data.head())

data.drop(columns=['Country'],axis=1,inplace=True)
print('Dataset contains null values: ',data.isnull().values.any())
print(data.info())
data['Origin']=data['Origin'].astype('category')

data['Month']=data['Month'].astype('category')

data['Origin'].replace({'Iran (Islamic Rep. of)':'Iran','Syrian Arab Rep.':'Syria','Russian Federation':'Russia','Dem. Rep. of the Congo':'Congo Rep. '},inplace=True)

data.info()

data.describe(include='all')
print('Number of incoming groups of asylum seekers per month')

print('-----------------------------------------------------')

print(pd.value_counts(data.Month))
print('Number of distinct countries in the sample : ',len(data.Origin.unique()))
print('Total groups of asylum seekers per origin')

print('-------------------------------------')

print(data['Origin'].value_counts()[:10])

print('Total asylum seekers per origin country')

print('---------------------------------------')

print(data[['Origin','Value']].groupby('Origin').sum().sort_values(by='Value',ascending=False)[1:10])

total_values=data.groupby('Origin').sum().drop('Year',axis=1)

total_values.sort_values(by='Value',inplace=True,ascending=False)



matplotlib.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 2, figsize=(25, 10))

sns.countplot(data['Origin'],order=pd.value_counts(data['Origin']).iloc[:20].index,palette="Blues_d",ax=axs[0])

sns.barplot(x=total_values[0:20].index,y=total_values['Value'][0:20],palette="Blues_d",ax=axs[1])

axs[0].set_title('20 countries with highiest count of groups of asylum seekers',fontsize=16 )

axs[1].set_title('Total asylum seekers, top 10 countries',fontsize=16)



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(size=14,rotation=45)

    plt.yticks(size=14,)

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(25, 15))

data['Origin'].value_counts()[:10].plot(kind='pie',ax=axs[0])

data.Value.groupby(data.Origin).sum().sort_values(ascending=False)[:10].plot(kind='pie',ax=axs[1])

axs[0].set_title('Count of groups',fontsize=16)

axs[0].set_ylabel('')

axs[1].set_title('Total asylum seekers',fontsize=16)

axs[1].set_ylabel('')

plt.show()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

world['centroid_col']=world.centroid

origin_counts=data['Origin'].value_counts()

top_origin_counts=origin_counts.nlargest(10)

top_countries=world[world['name'].isin(list(top_origin_counts.keys()))]

all_countries=world.loc[world['name'].isin(list(data['Origin']))]

all_countries.reset_index(drop=True, inplace=True)

all_countries.set_index('name',inplace=True)

for country in all_countries.index:

    all_countries.loc[country,'sum']=total_values.loc[country].values[0]
fig, ax = plt.subplots(figsize=(25,15))

plt.title('Total asylum seekers origin countries heatmap ( all countries accounted)',fontsize=16)

world.plot(ax=ax, color='white', edgecolor='black')

all_countries.plot(column='sum',legend='True',ax=ax, edgecolor='black',cmap='Reds')

plt.show()
top_countries=world[world['name'].isin(list(origin_counts.nlargest(20).keys()))]



map=folium.Map([37.993670, 23.731474], zoom_start=3)

folium.Marker([37.993670, 23.731474], popup='<i>Greece</i>',icon=folium.Icon(color='green', icon='home')).add_to(map)



for index, row in all_countries.iterrows():

   folium.Circle(

      location=[row['centroid_col'].y,row['centroid_col'].x],

      tooltip=row.name,

      popup=row['sum'],

      radius=row['sum']*12,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(map)

map
plt.figure(figsize=(22,11))

sns.lineplot(np.arange(2000,2019,1),data.groupby('Year').sum()['Value'])

plt.grid()

plt.xticks(np.arange(2000,2019,1),fontsize=14)

plt.yticks(np.arange(0,70000,5000),fontsize=14)

plt.title('Total sum of asylum seekers for each year',fontsize=16)

plt.axvline(x=2014,color='r',linestyle='--')

plt.text(2013.7,40000,'Beginning of refugee crisis',rotation=90,fontsize=14)

plt.grid()

plt.show()
plt.figure(figsize=(25,11))

order=['January','February','March','April','May','June','July','August','September','October','November','December']

sns.scatterplot(data.Year,data.Value,hue=data.Month,hue_order=order,palette=sns.color_palette("hls", 12))

plt.xticks(np.arange(2000,2019,1),fontsize=14)

plt.yticks(np.arange(0,4000,500),fontsize=14)

plt.title('Asylum seekers 2000-2018 for each year and per month ',fontsize=16)

plt.show()

top_data=data[data.Origin.isin(total_values[:10].index)]

plt.figure(figsize=(22,12))

plt.title('Asylum seekers for each year. Top 10 countries',fontsize=16)

sns.lineplot(top_data.Year,top_data.Value,hue=top_data.Origin,hue_order=total_values[:10].index)

plt.xticks(np.arange(2000,2019,1),fontsize=14)

plt.yticks(np.arange(0,4000,500),fontsize=14)

plt.xlabel('Year',fontsize=14)

plt.ylabel('Asylum seekers',fontsize=14)

plt.axvline(x=2014,color='r',linestyle='--')

plt.text(2013.7,2000,'Beginning of refugee crisis',rotation=90,fontsize=14)

plt.show()
group1=data[data['Origin'].isin(['Afghanistan','Pakistan','Iraq'])]

plt.figure(figsize=(22,12))

plt.title('Total asylum seekers from Pakistan,Iraq and Afghanistan',fontsize=16)

sns.lineplot(group1.Year,group1.Value,hue=group1.Origin,hue_order=['Pakistan','Iraq','Afghanistan'])

plt.xlabel('Year',fontsize=14)

plt.ylabel('Asylum seekers',fontsize=14)

plt.xticks(np.arange(2000,2019,1),fontsize=14)

plt.yticks(np.arange(0,4000,500),fontsize=14)



plt.axvline(x=2014,color='r',linestyle='--')

plt.text(2013.7,1800,'Beginning of refugee crisis',rotation=90,fontsize=14)

plt.show()
data['period']=data['Year']

data.loc[data['Year']<2014,'period']='before'

data.loc[data['Year']>=2014,'period']='after'



data['period']=data['period'].astype('category')

data['period'].cat.reorder_categories(['before', 'after'], inplace=True)

period_sums=data.groupby('period')['Value'].sum()
print('Asylum seekers groups count:')

print(data['period'].value_counts())

print('------------------------')

print('Total asylum seekers:')

print(period_sums)



fig, axs = plt.subplots(1, 2, figsize=(21, 13))

sns.barplot(period_sums.index,period_sums.values,ax=axs[0])

plt.suptitle('Asylum seekers before and after the refugee crisis of 2014',fontsize=16)

axs[0].set_title('Total asylum seekers ',fontsize=14)

axs[0].set_xticklabels(['before 2014','after 2014'],fontsize=14)

order=['January','February','March','April','May','June','July','August','September','October','November','December']

sns.countplot(data['period'],ax=axs[1],hue=data['Month'],hue_order=order)

axs[1].set_title('Number of groups',fontsize=14)

axs[1].set_xticklabels(['before 2014','after 2014'],fontsize=14)

plt.show()



top_data=data[data['Origin'].isin(total_values[:10].index)]

top_data.reset_index(drop=True, inplace=True)
plt.figure(figsize=(24,12))

sns.boxplot(x="Origin",y="Value",hue="period",data=top_data)

plt.title('Boxplots of asylum seekers before and after 2014',fontsize=16)

plt.xlabel('Origin',fontsize=14)

plt.yticks(np.arange(0,4000,500),fontsize=14)

plt.ylabel('Asylum seekers',fontsize=14)

plt.show()
plt.figure(figsize=(22,10))

sns.boxplot(x="Month",y="Value",hue="period",data=top_data,order=order, linewidth=2.5)

plt.title('Boxplots of asylum seekers for each month, before and after 2014',fontsize=16)

plt.xlabel('Months',fontsize=14)

plt.ylabel('Asylum seekers',fontsize=14)

plt.yticks(np.arange(0,4000,500),fontsize=14)



plt.show()
