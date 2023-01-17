# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
try:
    file = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')
# Filtering country name

file_spain = file[file.country_txt == 'Spain']
# Rows and columns

file_spain.shape
# Selecting columns

file_spain = file_spain[['iyear','imonth','iday','country_txt','region_txt','provstate','city',
                         'latitude','longitude','attacktype1_txt','targtype1_txt','target1',
                         'gname','nperps','weaptype1_txt','nkill','nwound','summary','motive']]
# Renaming columns

file_spain.columns = ['Year','Month','Day','Country','Region','Province','City','Latitude',
                      'Longitude','Attacktype','Target_General','Target','Group','Number_Terrorist',
                     'Weapon','Killed','Wounded','Summary','Motive']
# Checking Null Values 
# Total Rows: 3249

file_spain.isnull().sum()
# Data Cleaning

file_spain['Killed'].fillna(0, inplace = True)
file_spain['Wounded'].fillna(0, inplace = True)
# Sample Filtered

file_spain.head()
print('City with Highest Terrorist Attacks in Spain:', file_spain['City'].value_counts().index[0])
print('Province with Highest Terrorist Attacks in Spain:', file_spain['Province'].value_counts().index[0])
print('Group with Highest Terrorist Attacks in Spain:', file_spain['Group'].value_counts().index[0])
print('Maximum People Killed in a Terrorist Attack in Spain:', file_spain['Killed'].sort_values(ascending = False).iloc[0], 'Group:', file_spain.loc[file_spain['Killed'].idxmax(),'Group'])
file_spain.Year.plot(kind = 'hist', grid = True, color = 'green',bins = range(1970,2018), figsize = (14,6),alpha = 0.6)
plt.xticks(range(1970,2018),rotation = 45, fontsize = 10)
plt.yticks(range(0,300,25), fontsize = 10)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Number of Terrorist Attacks',fontsize = 12)
plt.title('Number of Terrorist Attacks By Year', fontsize = 14)
plt.xlim((1970,2018))
plt.show()
# Another way to develope this plot
# Note: years with no data have been removed

plt.subplots(figsize=(14,6))
sns.countplot(file_spain['Year'], data=file_spain, palette='RdYlGn_r', edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
file_spain.Weapon.value_counts().drop('Unknown').plot(kind = 'bar', figsize = (14,6), grid = True )
#plt.xlabel('Kind of Weapons', fontsize = 12)
plt.ylabel('Number of Terrorist Attacks',fontsize = 12)
plt.xticks(np.arange(7),('Explosives','Firearms','Incendiary','Melee','Fake Weapons','Vehicle','Sabotage Equipment'), rotation = 0, fontsize=10)
plt.title('Weapons', fontsize = 14)
plt.show()
killed_by_groups = file_spain.groupby('Group').Killed.agg(['sum','max'])
most_killed_by_group = killed_by_groups[killed_by_groups['sum'] > 0].sort_values('sum', ascending = False)
most_killed_by_group['sum'].drop('Unknown').head(10).plot(kind = 'bar', figsize = (14,6), grid = True)
plt.xticks(np.arange(10),('ETA','AL-QAIDA','GRAPO','MUSLIM EXTREMISTS','HEZBOLLAH','AAA','FRAP','BBE','GALICIAN GUERRILLA ARMY','NEO-FASCISTS'),rotation = 90, fontsize = 8)
plt.yticks(fontsize = 10)
#plt.xlabel('Group', fontsize = 12)
plt.ylabel('Number of Killed People',fontsize = 12)
plt.title('Most Killed People By Group', fontsize = 14)
plt.show()
file_spain.City.value_counts().drop('Unknown').head(10).plot(kind = 'bar', grid = True, figsize = (14,6))
plt.xticks(rotation = 90, fontsize = 10)
plt.yticks(fontsize = 10)
#plt.xlabel('Cities', fontsize = 12)
plt.ylabel('Number of Terrorist Attacks',fontsize = 12)
plt.title('Most Targeted Cities', fontsize = 14)
plt.show()
eta = file_spain[file_spain['Group'] == 'Basque Fatherland and Freedom (ETA)']
eta.head()
# We need Data over 2010
more_data= pd.Series({2011:0,2012:0,2013:0,2014:0,2015:0,2016:0,2017:0,2018:0})
eta.groupby(['Year']).Killed.sum().append(more_data).plot(kind = 'line', grid = True, figsize=(14,6))
eta.groupby(['Year']).Wounded.sum().append(more_data).plot(kind = 'line', grid = True, figsize=(14,6))
plt.xticks(range(1970,2019),fontsize = 10,rotation = 90)
plt.yticks(range(0,200,10),fontsize = 10)
#plt.xlabel( fontsize = 12)
plt.xlim((1970,2018))
plt.ylabel('Number of Killed People',fontsize = 12)
plt.title('Killed and Wounded People by ETA', fontsize = 14)
plt.legend(['Killed','Wounded'])
plt.show()
import plotly.plotly as py
eta = eta[eta.Latitude.notnull() & eta.Killed > 0]
eta['text'] = "City : " + eta["City"].astype(str) + " <br>"+"Year : " + eta['Year'].astype(str) +\
                 " <br>" + "Killed : " + (eta["Killed"].astype(int)).astype(str) +\
                 " <br>" + "Attacktype : " + eta["Attacktype"]
attacks = dict(
               type = 'scattergeo',
               lon = eta['Longitude'],
               lat = eta['Latitude'],
               text = eta['text'],
               hoverinfo = 'text',
               mode = 'markers',
               marker = dict(
                     size = eta["Killed"] ** 0.25 * 10,
                     opacity = 0.7,
                     color = 'rgb(10, 160, 200)'
               )
         )
        
layout = dict(
            title = 'ETA Attacks in Spain',
            hovermode='closest',
            geo = dict(
                
                showframe=True,
                showland=True,
                landcolor = 'rgb(210, 210, 210)',
                showcountries = True,
                lonaxis = dict( range= [ -10.0, 5.0 ] ),
                lataxis = dict( range= [ 35.0, 45.0 ] ),
            )
         )
figure = dict(data = [attacks], layout = layout)
py(figure)
plt.figure(figsize=(20,20))
europe = file[file["region_txt"].isin(["Eastern Europe", "Western Europe"])]

EU = Basemap(projection='mill', llcrnrlat = 30, urcrnrlat = 75, llcrnrlon = -15, urcrnrlon = 70, resolution = 'l')
EU.drawcoastlines(color='black')
#EU.etopo()
EU.drawcountries(color='orange')
#EU.drawstates(color='green')
#EU.drawrivers()

x, y = EU(list(europe["longitude"].astype("float")), list(europe["latitude"].astype(float)))
EU.plot(x, y, "bo", markersize = 2, alpha = 0.6, color = 'blue')

plt.title('Terror Attacks on Europe (1970-2018)')
plt.show()
plt.figure(figsize = (20,20))
SP = Basemap(projection='mill', llcrnrlat = 34, urcrnrlat = 45, llcrnrlon = -12, urcrnrlon = 5, resolution = 'l')
SP.drawcoastlines(color='black')
#SP.etopo()
SP.drawcountries(color='black')
#SP.drawstates(color='green')
#SP.drawrivers()
SP.fillcontinents(color='grey',alpha = 0.2)
x , y = SP(list(file_spain['Longitude']),list(file_spain['Latitude']))

SP.plot(x,y,"bo", markersize = 5, alpha = 0.6, color = 'orange')
plt.title('Terror Attacks on Spain (1970-2016)')
plt.show()
eta['Weapon'].value_counts().plot(kind='pie',figsize=(10,10))
plt.title('ETA Prefer Weapons')
plt.show()
