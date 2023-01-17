# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
terrorism = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
terrorism.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terrorism=terrorism[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terrorism['Region'].value_counts()
plt.subplots(figsize=(15,6))
sns.countplot('Region',data=terrorism,palette='gnuplot2_r',edgecolor=sns.color_palette('dark',7),order=terrorism['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year in the World By Region')
plt.show()
terror_region=pd.crosstab(terrorism.Year,terrorism.Region)
terror_region.plot(color=sns.color_palette('terrain_r',12))
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.show()
east = terrorism[terrorism.Region == "Middle East & North Africa"]
east.head(5)
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=east,palette='afmhot_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities in the Middle East and North Africa Each Year')
plt.show()
east.Year.value_counts()
plt.subplots(figsize=(15,6))
sns.countplot('Country',data=east,palette='hot_r',edgecolor=sns.color_palette('dark',7),order=east['Country'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities in each Country in the Middle East and North Africa Each Year')
plt.show()
east.Country.value_counts()
east.city.value_counts().sort_values(ascending=False).head(10)
cities = east.city.dropna(False)
plt.subplots(figsize=(14,12))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.imsave(arr = wordcloud, fname = 'wordcloud.png')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=east,palette='Spectral_r',order=east['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists in the Middle East and North Africa')
plt.show()

pd.crosstab(east.Country,east.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('CMRmap_r',9))
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()
c_terror=east['Country'].value_counts()[:15].to_frame()
c_terror.columns=['Attacks']
c_kill=east.groupby('Country')['Killed'].sum().to_frame()
c_terror.merge(c_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(terrorism['Target_type'],palette='inferno',order=terrorism['Target_type'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Favorite Targets in MENA')
plt.show()
sns.barplot(east['Group'].value_counts()[1:15].values,east['Group'].value_counts()[1:15].index,palette=('gist_heat_r'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.title('Terrorist Groups with Highest Terror Attacks in MENA')
plt.show()

top10=east[east['Group'].isin(east['Group'].value_counts()[1:11].index)]
pd.crosstab(top10.Year,top10.Group).plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('Month',data=east,palette='hot_r',edgecolor=sns.color_palette('dark',7),order=east['Month'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities in each Month in MENA')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('Day',data=east,palette='hot_r',edgecolor=sns.color_palette('dark',7),order=east['Day'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities in each Day in MENA')
plt.show()
motives = east.Motive.dropna(False)
plt.subplots(figsize=(14,12))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(motives))
plt.axis('off')
plt.imshow(wordcloud)
plt.imsave(arr = wordcloud, fname = 'wordcloud.png')
plt.show()
t_ıraq=east[east['Country']=='Iraq']
t_ıraq_f=t_ıraq.copy()
t_ıraq_f.dropna(subset=['latitude','longitude'],inplace=True)
location_ıraq=t_ıraq_f[['latitude','longitude']]
city_ıraq=t_ıraq_f['city']
killed_ıraq=t_ıraq_f['Killed']
wound_ıraq=t_ıraq_f['Wounded']
target_ıraq=t_ıraq_f['Target_type']

f,ax=plt.subplots(1,2,figsize=(25,12))
ıraq_groups=t_ıraq['Group'].value_counts()[1:11].index
ıraq_groups=t_ıraq[t_ıraq['Group'].isin(ıraq_groups)]
sns.countplot(y='Group',data=ıraq_groups,ax=ax[0])
sns.countplot(y='AttackType',data=t_ıraq,ax=ax[1])
plt.subplots_adjust(hspace=0.3,wspace=0.6)
ax[0].set_title('Top Terrorist Groups in IRAQ')
ax[1].set_title('Favorite Attack Types in IRAQ')
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
plt.show()
t_turkey=east[east['Country']=='Turkey']
t_turkey_f=t_turkey.copy()
t_turkey_f.dropna(subset=['latitude','longitude'],inplace=True)
location_turkey=t_turkey_f[['latitude','longitude']]
city_turkey=t_turkey_f['city']
killed_turkey=t_turkey_f['Killed']
wound_turkey=t_turkey_f['Wounded']
target_turkey=t_turkey_f['Target_type']

f,ax=plt.subplots(1,2,figsize=(25,12))
turkey_groups=t_turkey['Group'].value_counts()[1:11].index
turkey_groups=t_turkey[t_turkey['Group'].isin(turkey_groups)]
sns.countplot(y='Group',data=turkey_groups,ax=ax[0])
sns.countplot(y='AttackType',data=t_turkey,ax=ax[1])
plt.subplots_adjust(hspace=0.3,wspace=0.6)
ax[0].set_title('Top Terrorist Groups in TURKEY')
ax[1].set_title('Favorite Attack Types in TURKEY')
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
plt.show()
