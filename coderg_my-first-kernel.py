# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/globalterrorismdb_0617dist.csv",encoding='ISO-8859-1')

data.info()


data.columns

data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


data.head(5)
plt.figure(figsize=(15,15))
data.iday.plot(kind="line",color="r",label="Days",linewidth=0.5,alpha=0.8, grid=True,linestyle=":")
data.imonth.plot(color="g",label="Month",linewidth=0.5,alpha=0.8,grid=True,linestyle="-.")
#plt.xticks( [0,88888])

plt.legend(loc="upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("")
plt.show()


data.plot(kind='scatter',x='region',y='targtype1', alpha = 0.5,color = 'magenta')
plt.xlabel('region')
plt.ylabel('target type id')
plt.title('Region Target Type id Scatter Plot')
plt.show()

data.head()

data.iyear.plot(kind='hist', bins=50,figsize=(5,8))
plt.show()
data.country.plot(kind='hist', bins=50,figsize=(5,8))
plt.show()
data.attacktype1.plot(kind='hist', bins=50,figsize=(5,8))

data_fr=data['iyear']
data[(data_fr>1990) & (data['city']=="Istanbul")& (data['attacktype1']==3)]
m3 = Basemap(projection='mill',llcrnrlat=15,urcrnrlat=60, llcrnrlon=-20,urcrnrlon=60,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
data['casualities']=data['nkill']+data['nwound']
lat_100=list(data[data['casualities']>=150].latitude)
long_100=list(data[data['casualities']>=150].longitude)
x_100,y_100=m3(long_100,lat_100)
m3.plot(x_100, y_100,'go',markersize=10,color = 'r')
lat_=list(data[data['casualities']<150].latitude)
long_=list(data[data['casualities']<150].longitude)
x_,y_=m3(long_,lat_)
m3.plot(x_, y_,'go',markersize=2,color = 'yellow',alpha=0.1)
m3.drawcoastlines()
m3.drawcountries()
m3.fillcontinents(lake_color='blue')
m3.drawmapboundary(fill_color='aqua')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.title('Turkey  Terrorist Attacks')
plt.legend(loc='lower left',handles=[mpatches.Patch(color='yellow', label = "< 150 casualities"),
                    mpatches.Patch(color='red',label='> 150 casualities')])
plt.show()
for index,value in data[['region']].iterrows():
    print(index," : ",value)