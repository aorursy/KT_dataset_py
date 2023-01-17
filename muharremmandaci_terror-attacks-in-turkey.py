# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
# I got help from someone else's kernel. (https://www.kaggle.com/ash316/terrorism-around-the-world/notebook)
terror_data=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
pd.options.display.max_columns = 135
terror_data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','city':'City','latitude':'Latitude','longitude':'Longitude','success':'Success','attacktype1_txt':'AttackType','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','weaptype1_txt':'WeaponType','targtype1_txt':'TargetType','motive':'Motive','targsubtype1_txt':'TargetSubType','corp1':'CorporationName','target1':'TargetName','natlty1':'NationalityOfTarget'},inplace=True)
terror_data=terror_data[['Year','Month','Day','Country','Region','City','Latitude','Longitude','Success','AttackType','Killed','Wounded','Summary','Group','WeaponType','Motive','TargetType','TargetSubType','CorporationName','TargetName','NationalityOfTarget']]
terror_data['casualities']=terror_data['Killed']+terror_data['Wounded']
terror_data.head(3)
terror_data.info()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(terror_data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
print("Top 3 countries with Highest Terrorist Attacks (Top 3)")
for index in range(3):
    print(terror_data['Country'].value_counts().index[index], " : ", terror_data['Country'].value_counts()[index], "times")
print("Top 3 regions with Highest Terrorist Attacks (Top 3)")
for index in range(3):
    print(terror_data['Region'].value_counts().index[index], " : ", terror_data['Region'].value_counts()[index], "times")
#print("Top 3 years with Highest Terrorist Attacks (Top 3)")
#for index in range(3):
#    print(terror_data['Year'].value_counts().index[index], " : ", terror_data['Year'].value_counts()[index], "times")
turkey_data = terror_data[terror_data["Country"]=="Turkey"]
turkey_data.head(5)
plt.subplots(figsize=(18,9))
sns.countplot('Year',data=turkey_data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year in Turkey')
plt.show()
plt.subplots(figsize=(18,9))
sns.countplot('Month',data=turkey_data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Month in Turkey')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=turkey_data,palette='inferno',order=turkey_data['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists in Turkey')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('WeaponType',data=turkey_data,palette='inferno',order=turkey_data['WeaponType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Using Weapon Types by Terrorists in Turkey')
plt.show()
sns.barplot(turkey_data['CorporationName'].value_counts()[0:15].values,turkey_data['CorporationName'].value_counts()[0:15].index,palette=('inferno'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.title('Counts of Terror Attacks for Corporations')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(turkey_data['TargetType'],palette='inferno',order=turkey_data['TargetType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Favorite Targets')
plt.show()
city_terror=turkey_data['City'].value_counts()[:15].to_frame()
city_terror.columns=['Attacks']
city_kill=turkey_data.groupby('City')['Killed'].sum().to_frame()
city_terror.merge(city_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
sns.barplot(turkey_data['Group'].value_counts()[0:15].values,turkey_data['Group'].value_counts()[0:15].index,palette=('inferno'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.title('Terrorist Groups with Highest Terror Attacks in Turkey')
plt.show()
top_groups10=turkey_data[turkey_data['Group'].isin(turkey_data['Group'].value_counts()[0:10].index)]
pd.crosstab(top_groups10.Year,top_groups10.Group).plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()