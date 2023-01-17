# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import folium.plugins







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1')



data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country',

                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target',

                       'nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group',

                       'targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)





data.head(5)
data.info()

data.dtypes





# shape gives number of rows and columns in a tuble

data.shape
data.corr()
f,ax=plt.subplots(figsize=(30,30))

sns.heatmap(data.corr(),annot=True,linewidth=2,fmt='.1f',ax=ax)

plt.show()
print('Country with Highest Terrorist Attacks:',data['Country'].value_counts().index[0])

print('Regions with Highest Terrorist Attacks:',data['Region'].value_counts().index[0])
plt.subplots(figsize=(15,6))

sns.countplot('Year',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities Each Year')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot(data['Target_type'],palette='inferno',order=data['Target_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Favorite Targets')

plt.show()
pd.crosstab(data.Region,data.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
sns.barplot(data['Group'].value_counts()[1:15].values,data['Group'].value_counts()[1:15].index,palette=('inferno'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Terrorist Groups with Highest Terror Attacks')

plt.show()
terror_turkey=data[data['Country']=='Turkey']

terror_turkey_fol=terror_turkey.copy()

terror_turkey_fol.dropna(subset=['latitude','longitude'],inplace=True)

location_ind=terror_turkey_fol[['latitude','longitude']][:5000]

city_ind=terror_turkey_fol['city'][:5000]

killed_ind=terror_turkey_fol['Killed'][:5000]

wound_ind=terror_turkey_fol['Wounded'][:5000]

target_ind=terror_turkey_fol['Target_type'][:5000]







f,ax=plt.subplots(1,2,figsize=(25,12))

ind_groups=terror_turkey['Group'].value_counts()[1:11].index

ind_groups=terror_turkey[terror_turkey['Group'].isin(ind_groups)]

sns.countplot(y='Group',data=ind_groups,ax=ax[0])

ax[0].set_title('Top Terrorist Groups')

sns.countplot(y='AttackType',data=terror_turkey,ax=ax[1])

ax[1].set_title('Favorite Attack Types')

plt.subplots_adjust(hspace=0.3,wspace=0.6)

ax[0].tick_params(labelsize=15)

ax[1].tick_params(labelsize=15)

plt.show()
data.Year.plot(kind="line",color="green",label="Purchase",linewidth=50,alpha=0.5,grid=True,linestyle=":")

data.country .plot(kind="line",color="red",label="User_ID",linewidth=50,alpha=0.5,grid=True,linestyle=":")

plt.legend(loc='upper right')

plt.xlabel('x Year')

plt.ylabel('y country')

plt.title('Line Plot')

plt.show()
data.plot(kind="scatter",x='Year',y='country',alpha=0.05,color="red")

plt.xlabel('Year')

plt.ylabel('Country')

plt.title('scatter plot')

plt.show()
data.Year.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
x= data['Country']=='Turkey'

data[x]

y=data['Year']> 1970

data[y].head(3)
print(data['Target_type'].value_counts(dropna =False))
print(data['AttackType'].value_counts(dropna =False))
data.describe()
data.boxplot(column='Year',by = 'AttackType',figsize=(17,8))

plt.show()
melted = pd.melt(frame=data,id_vars = 'Year', value_vars= ['Target_type','AttackType'])

melted
data_Target_type= data['Target_type'].head(15)

data_AttackType= data['AttackType'].head(15)

conc_data = pd.concat([data_Target_type,data_AttackType],axis =1) 

conc_data
data = data.loc[:,["Target_type","AttackType","Year"]]

data.plot()

plt.show()
# Setting index : type 1 is outer type 2 is inner index

data = data.set_index(["Target_type","AttackType"]) 

data.head(10)


