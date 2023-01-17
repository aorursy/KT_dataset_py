

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#importing the libraries needed for visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.patches as mpatches
from matplotlib import animation, rc
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import folium
import folium.plugins
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
!pip install -q scipy
import io
import base64
import codecs
import warnings
warnings.filterwarnings('ignore')
from IPython.display import HTML, display

df = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df.head(10)
df.shape
df.describe()
df.columns
df['casualties']=df['nkill']+df['nwound']
df['country_txt'].value_counts()
df['region_txt'].value_counts()
df['gname'].value_counts()
df['city'].value_counts()
df['attacktype1_txt'].value_counts()
df['targtype1_txt'].value_counts()
df['targsubtype1_txt'].value_counts()
print('Country with Highest Terrorist Attacks:',df['country_txt'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',df['region_txt'].value_counts().index[0])
print('Maximum people killed in an attack are:',df['nkill'].max(),'that took place in',df.loc[df['nkill'].idxmax()].country_txt)
df['iyear'].value_counts()
df['natlty1_txt'].value_counts()
print("Nationality of the maximally targetted group is:", df['natlty1_txt'].value_counts().index[0])
plt.subplots(figsize=(15,8))
sns.countplot('iyear',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.xlabel('Year of attack')
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('attacktype1_txt',data=df,palette='inferno',order=df['attacktype1_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Attack Type')
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('targtype1_txt',data=df,palette='inferno',order=df['targtype1_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Target Type')
plt.title('Target by Terrorists')
plt.show()
plt.subplots(figsize=(20,5))
sns.countplot('targsubtype1_txt',data=df,palette='inferno',order=df['targsubtype1_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Sub Target Type')
plt.title('Sub Targets by Terrorists')
plt.show()
print("The top sub targets of terrorists are:", df['targsubtype1_txt'].value_counts().index[:5])
plt.subplots(figsize=(15,5))
sns.countplot('region_txt',data=df,palette='inferno',order=df['region_txt'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Regions')
plt.title('Number Of Terrorist Activities By Region')
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('country_txt',data=df,palette='inferno',order=df['country_txt'].value_counts()[:15].index)
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.title('Number Of Terrorist Activities By Countries')
plt.show()
df_region=pd.crosstab(df.iyear,df.region_txt)
df_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(13,6)
plt.show()
df_region=pd.crosstab(df.iyear,df.targtype1_txt)
df_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(13,6)
plt.show()
df['casualties'].sum()
df['nkill'].sum()
df['nwound'].sum()
df1=df.groupby('region_txt')['casualties'].sum()
df2=df1.to_frame()
df2.rename(columns = {"" : "Casualties"}, inplace = True)
df2.insert(1,'Year',df.iyear,True)
df2.reset_index()

df.groupby(['region_txt', 'iyear'])['casualties'].count()
plt.subplots(figsize=(18,8))
df.groupby(['region_txt', 'iyear']).count()['casualties'].plot()
#sns.countplot('country_txt',data=df,palette='inferno',order=df['country_txt'].value_counts()[:15].index)
plt.xticks(rotation=90)
#plt.xlabel('Countries')
plt.ylabel("Casualties")
#plt.title('Number Of Terrorist Activities By Countries')
#plt.show()

df.groupby(['region_txt', 'iyear']).count()['casualties'].unstack('region_txt').plot(figsize=(18,8))
plt.xticks(rotation=90)

plt.ylabel("Casualties")
d=df.groupby(['region_txt', 'iyear'])['casualties'].sum()
d
d=df.groupby(['iyear','region_txt'])['nkill'].sum()
plot_df = d.unstack('region_txt').loc[:]
plot_df.index = pd.PeriodIndex(plot_df.index.tolist(),freq='A')
plot_df.plot(figsize=(15,8),color=sns.color_palette('Set2',12))
plt.xlabel("Year")
plt.ylabel("Killed")
#ax.show()
d=df.groupby(['iyear','region_txt'])['nwound'].sum()
plot_df = d.unstack('region_txt').loc[:]
plot_df.index = pd.PeriodIndex(plot_df.index.tolist(),freq='A')
plot_df.plot(figsize=(15,8),color=sns.color_palette('Set2',12))
plt.xlabel("Year")
plt.ylabel("Wounded")
d=df.groupby(['iyear','region_txt'])['casualties'].sum()
plot_df = d.unstack('region_txt').loc[:]
plot_df.index = pd.PeriodIndex(plot_df.index.tolist(),freq='A')
plot_df.plot(figsize=(15,8),color=sns.color_palette('Set2',12))
plt.xlabel("Year")
plt.ylabel("Casualties")
count=df['country_txt'].value_counts()[:15].to_frame()
count.columns=['Attacks']
data=df.groupby('country_txt')['nkill'].sum().to_frame()
count.merge(data,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
groups=df[df['gname'].isin(df['gname'].value_counts()[:14].index)]
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(lake_color='aqua')
m.drawmapboundary(fill_color='aqua')
fig=plt.gcf()
fig.set_size_inches(22,10)
colors=['r','g','b','y','#800000','#ff1100','#8202fa','#20fad9','#ff5733','#fa02c6',"#f99504",'#b3b6b7','#8e44ad','#1a2b3c']
group=list(groups['gname'].unique())
def group_point(group,color,label):
    lat_group=list(groups[groups['gname']==group].latitude)
    long_group=list(groups[groups['gname']==group].longitude)
    x_group,y_group=m(long_group,lat_group)
    m.plot(x_group,y_group,'go',markersize=3,color=j,label=i)
for i,j in zip(group,colors):
    group_point(i,j,i)
legend=plt.legend(loc='lower left',frameon=True,prop={'size':10})
frame=legend.get_frame()
frame.set_facecolor('white')
plt.title('Terrorist Groups')
plt.show()
groups=df[df['region_txt'].isin(df['region_txt'].value_counts()[:14].index)]
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(lake_color='black')
m.drawmapboundary(fill_color='white')
fig=plt.gcf()
fig.set_size_inches(22,10)
colors=['r','g','b','y','#800000','#ff1100','#8202fa','#20fad9','#ff5733','#fa02c6',"#f99504",'#b3b6b7','#8e44ad','#1a2b3c']
group=list(groups['region_txt'].unique())
def group_point(group,color,label):
    lat_group=list(groups[groups['region_txt']==group].latitude)
    long_group=list(groups[groups['region_txt']==group].longitude)
    x_group,y_group=m(long_group,lat_group)
    m.plot(x_group,y_group,'go',markersize=3,color=j,label=i)
for i,j in zip(group,colors):
    group_point(i,j,i)
legend=plt.legend(loc='lower left',frameon=True,prop={'size':14})
frame=legend.get_frame()
frame.set_facecolor('white')
plt.title('Target areas of terrorist attacks')
plt.show()
groups=df[df['country_txt'].isin(df['country_txt'].value_counts()[:14].index)]
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(lake_color='black')
m.drawmapboundary(fill_color='white')
fig=plt.gcf()
fig.set_size_inches(22,10)
colors=['r','g','b','y','#800000','#ff1100','#8202fa','#20fad9','#ff5733','#fa02c6',"#f99504",'#b3b6b7','#8e44ad','#1a2b3c']
group=list(groups['country_txt'].unique())
def group_point(group,color,label):
    lat_group=list(groups[groups['country_txt']==group].latitude)
    long_group=list(groups[groups['country_txt']==group].longitude)
    x_group,y_group=m(long_group,lat_group)
    m.plot(x_group,y_group,'go',markersize=3,color=j,label=i)
for i,j in zip(group,colors):
    group_point(i,j,i)
legend=plt.legend(loc='lower left',frameon=True,prop={'size':16})
frame=legend.get_frame()
frame.set_facecolor('white')
plt.title('Target countries of terrorist attacks')
plt.show()
fig = plt.figure(figsize = (10,6))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Animation Of Terrorist Activities'+'\n'+'Year:' +str(Year))
    m6 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    lat6=list(df[df['iyear']==Year].latitude)
    long6=list(df[df['iyear']==Year].longitude)
    x6,y6=m6(long6,lat6)
    m6.scatter(x6, y6,s=[(kill+wound)*0.1 for kill,wound in zip(df[df['iyear']==Year].nkill,df[df['iyear']==Year].nwound)],color = 'r')
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(zorder = 1,alpha=0.4)
    m6.drawmapboundary()
ani = animation.FuncAnimation(fig,animate,list(df.iyear.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
