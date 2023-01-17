# importing all the necessary packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import matplotlib.patches as mpatches



import warnings

warnings.filterwarnings('ignore')
# importing the dataset

terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
# Preview the raw dataset

terror.head ()
# Renaming the features

terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
# Retaining only usefull features

terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
# Preview the cleaned dataset

terror.head ()
# Creating new feature "Casualities" by adding "Killed" and "Wounded" features

terror['casualities']=terror['Killed']+terror['Wounded']
terror.head (3)
terror.isnull().sum()
terror['Group'].value_counts().head (5)
print('Country with Highest Terrorist Attacks:',terror['Country'].value_counts().index[0])

print('Regions with Highest Terrorist Attacks:',terror['Region'].value_counts().index[0])

print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)

print('Maximum casualties of', terror['casualities'].max(), 'happened in a single attack in', terror.loc[terror['casualities'].idxmax()].Country)
plt.subplots(figsize=(15,5))

sns.countplot('Year',data=terror)

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities Each Year')

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot('Region',data=terror,order=terror['Region'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities Each Region')

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot('Target_type',data=terror,order=terror['Target_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Target victims')

plt.show()
m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180)

lat_100=list(terror[terror['casualities']<100].latitude)

long_100=list(terror[terror['casualities']<100].longitude)

x_100,y_100=m3(long_100, lat_100)

m3.drawcoastlines()

m3.drawcountries()

m3.plot(x_100, y_100,'go',markersize=0.5,color = 'g')

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.title("Terroist attacks with lesser than 100 casualities")
m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180)

lat_100=list(terror[terror['casualities']>=100].latitude)

long_100=list(terror[terror['casualities']>=100].longitude)

x_100,y_100=m3(long_100, lat_100)

m3.drawcoastlines()

m3.drawcountries()

m3.plot(x_100, y_100,'go',markersize=5,color = 'r')

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.title("Terroist attacks with more than 100 casualities")
terror_region=pd.crosstab(terror.Year,terror.Region)

terror_region.plot(color=sns.color_palette('Set2',12))

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()
terror_type = pd.crosstab(terror.Region,terror.AttackType)

terror_type
terror_type.plot.barh(stacked=True, width=1)

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
# Top 20 countries affected by terrorism

coun_terror=terror['Country'].value_counts()[:20].to_frame() # to_frame() function will generate a dataframe out of the results. 

coun_terror.columns=['Attacks']

coun_terror
coun_terror.plot.bar()

fig=plt.gcf()

fig.set_size_inches(18,6)
# This will give the number of people killed in every country collectively

coun_kill=terror.groupby('Country')['Killed'].sum().to_frame() 

coun_kill.head ()
# This will merge the coun_terror and coun_kill datasets 

# and give top 20 countries with no of attacks and no of people killed 

attack_kill = coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left')

attack_kill
# Plotting the same on a bar chart

attack_kill.plot.bar()

fig=plt.gcf()

fig.set_size_inches(18,6)
# To find which terrorist group is most active

coun_group=terror['Group'].value_counts()[:20].to_frame()

coun_group
coun_group[1:20].plot.bar()

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.title ("Most active terrorist groups")
top_groups10=terror[terror['Group'].isin(terror['Group'].value_counts()[1:11].index)]

noto_gro = pd.crosstab(top_groups10.Year,top_groups10.Group)

noto_gro.plot(color=sns.color_palette('Paired',10))

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()
# Favorite attacking style world wide

sns.countplot(terror['AttackType'], order = terror['AttackType'].value_counts().index)

plt.xticks(rotation=90)
terror_india=terror[terror['Country']=='India']
terror_india.head ()
m3 = Basemap(projection='mill', llcrnrlat=5,urcrnrlat=37,llcrnrlon=67,urcrnrlon=99)

lat_100=list(terror[terror['casualities']<100].latitude)

long_100=list(terror[terror['casualities']<100].longitude)

x_100,y_100=m3(long_100, lat_100)

m3.drawcoastlines()

m3.drawcountries()

m3.plot(x_100, y_100,'go',markersize=1,color = 'g')

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.title("Terroist attacks with lesser than 100 casualities")
m3 = Basemap(projection='mill', llcrnrlat=5,urcrnrlat=37,llcrnrlon=67,urcrnrlon=99)

lat_100=list(terror[terror['casualities']>=100].latitude)

long_100=list(terror[terror['casualities']>=100].longitude)

x_100,y_100=m3(long_100, lat_100)

m3.drawcoastlines()

m3.drawcountries()

m3.plot(x_100, y_100,'go',markersize=5,color = 'r')

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.title("Terroist attacks with lesser than 100 casualities")