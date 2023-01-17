import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cartopy.crs as ccrs
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import io
import base64
import codecs
from IPython.display import HTML,display
from mpl_toolkits.basemap import Basemap
import plotly.offline as py
py.init_notebook_mode(connected=True)
import os
import threading
def color_from_int(num,max_value):
    red = num/max_value
    blue = 1.0 - (num/max_value)
    green = 1.0 - (num/max_value)
    return (red, green, blue)

#Reading data
#consider following columns from whole database
columns = ["eventid","iyear","imonth","iday","approxdate","extended","resolution","country","country_txt","region","region_txt","provstate","city","latitude","longitude","specificity","summary","doubtterr","alternative","multiple","related","success","suicide","attacktype1_txt","natlty1_txt","natlty1_txt","weaptype1_txt","nkill","nwound","nkillter","nwoundte","property","propextent","propextent_txt"]
df_main = pd.read_csv("../input/globalterrorismdb_0617dist.csv",encoding = "ISO-8859-1",usecols=columns,low_memory=False) #,index_col=["eventid"]
df_main["victim_casualties"] = (df_main["nkill"] - df_main["nkillter"]) + (df_main["nwound"] - df_main["nwoundte"])
df_main["terrorist_casualties"] = df_main["nkillter"] + df_main["nwoundte"]
print("Total number of events records = ",df_main.shape[0])
display(df_main.sample(5))
#Basic data analysis
print("Number of events with no recorded country = ",len(df_main[np.isnan(df_main["country"])])) #0
print(df_main.isnull().sum())
df_temp = df_main[~np.isnan(df_main["victim_casualties"]) & ~np.isnan(df_main["terrorist_casualties"])]
#df_temp = df_temp[df_temp["terrorist_casualties"]!=0]
pd.options.mode.chained_assignment = None
df_temp["terrorist_casualties"] = df_temp["terrorist_casualties"].replace(to_replace=0,value=1) #to avoid division by zero 
print("Victim casualty to Terrorist casualty ratio for {0} events with available casualty records is {1}".format(df_temp.shape[0],np.round(df_temp["victim_casualties"]/df_temp["terrorist_casualties"]).mean()))
pd.options.mode.chained_assignment = 'warn'
#Countries with highest terrorist attacks 
#print(df_main["country"].max(),df_main["country"].min(),len(df_main["country"].unique())) #1004,4,205
df_temp = df_main.groupby(["country_txt"])["eventid"].count().reset_index(name="attack_count").sort_values("attack_count",ascending=False)
#display(df_temp.head())
min_val = df_temp.loc[0:15]["attack_count"].min()
max_val = df_temp.loc[0:15]["attack_count"].max()
plt.figure(figsize=(10,10))
plt.title("Countries with highest attack counts")
plt.bar(x=df_temp[0:15]["country_txt"],height=df_temp[0:15]["attack_count"],align='center',color=df_temp.loc[0:15]["attack_count"].apply(lambda x, max_val=max_val: color_from_int(x,max_val)))
plt.xticks(rotation='vertical')
plt.show()
#Year wise global terrorist attacks
print("We have data for years {0} to {1}".format(df_main["iyear"].min(),df_main["iyear"].max()))
df_temp = df_main.groupby("iyear")["eventid"].count().reset_index(name="attacks")#.sort_values("attacks",ascending=False)
#max_val = df_temp["attacks"].max()
#display(df_temp)
plt.figure(figsize=(10,10))
plt.title("Year wise attack counts")
plt.xlabel("Years")
plt.ylabel("Number of attacks")
plt.plot(df_temp["iyear"],df_temp["attacks"])
#plt.bar(x=df_temp["iyear"],height=df_temp["attacks"],align='center',color=df_temp["attacks"].apply(lambda x,max_val=max_val : color_from_int(x,max_val)))
#plt.xticks(rotation='vertical')
plt.show()
#30 years of terrorism on world map
fig = plt.figure(figsize=(12,50))
ax = fig.add_axes([0,0,1,1]) #figure size in 0-1 bottom-left to width-height
m = Basemap(projection='cyl',llcrnrlat=-90,llcrnrlon=-180,urcrnrlon=180,urcrnrlat=90,resolution=None)
m.shadedrelief()
#m.etopo()
lats_lons = df_main[~np.isnan(df_main["latitude"]) & ~np.isnan(df_main["longitude"])]
m.scatter(lats_lons["longitude"],lats_lons["latitude"],marker='o',c='r',s=10,latlon=True,alpha=0.5)
plt.title("Global terrorism (30 years) - Attacks")
plt.show(block=False)
#Property damage due to attacks
#1-catastophic, 2-major, 3-minor, 4-unknown
fig = plt.figure(figsize=(20,15))
ax = fig.add_axes([0,0,1,1])
lats_lons = lats_lons[(lats_lons["property"]==1) & (lats_lons["propextent"].notnull())]

lats_lons_minor = lats_lons[lats_lons["propextent"] == 3.0]
lats_lons_major = lats_lons[lats_lons["propextent"] == 2.0]
lats_lons_catas = lats_lons[lats_lons["propextent"] == 1.0]

count = 0
#line, = ax.plot([],[], '-')
#line.set_xdata(x[:i])
label_dict = {0:"Minor",1:"Minor",2:"Major",3:"Catastrophic"}
color_dict = {0:'yellow',1:'yellow',2:'darkorange',3:'red'}
size_dict = {0:15,1:15,2:20,3:30}

def plot_loop(lats_lons):
    global count
    global label_dict
    global color_dict
    global size_dict
    
    #t = count%3
    #print("called:",count,len(lats_lons))
    ax.clear()
    label = ""
    ax.set_title("Global terrorism Property Damage - {0}".format(label_dict[count]))
    m = Basemap(projection='cyl',llcrnrlat=-90,llcrnrlon=-180,urcrnrlon=180,urcrnrlat=90,resolution=None)
    m.shadedrelief()
    m.scatter(lats_lons["longitude"],lats_lons["latitude"],marker='o',c=color_dict[count],s=size_dict[count],latlon=True)
    count = count + 1

ani = mpl.animation.FuncAnimation(fig, plot_loop,frames=[lats_lons_minor,lats_lons_major,lats_lons_catas],interval=2000)
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close('all')
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
display(HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii'))))
#http://bagrow.com/dsv/heatmap_basemap.html
#Types of weapons used in attacks
df_temp = df_main.groupby("weaptype1_txt")["eventid"].count().reset_index(name="attacks").sort_values("attacks",ascending=False)
df_temp["weaptype1_txt"].replace("Vehicle [\(\)\.].*","Vehicle",regex=True,inplace=True)
max_val = df_temp["attacks"].max()
display(df_temp)
plt.figure(figsize=(10,10))
plt.title("Frequency of weapons used in attacks")
plt.bar(x=df_temp["weaptype1_txt"],height=df_temp["attacks"],align='center',color=df_temp["attacks"].apply(lambda x,max_val=max_val : color_from_int(x,max_val)))
plt.xticks(rotation='vertical')
plt.show()
#Which weapons have caused most damage (people and property)
df_temp = df_main.copy()
df_temp = df_temp[df_temp["nkill"] != 0]
df_temp = df_temp[(~df_temp["weaptype1_txt"].isna())]
#df_temp = df_temp[~df_temp["propextent"].isna()]
df_temp["weaptype1_txt"].replace("Vehicle [\(\)\.].*","Vehicle",regex=True,inplace=True)
df_temp = df_temp.groupby("weaptype1_txt").agg({"nkill":np.sum,"nwound":np.sum})
df_temp["casualty"] = df_temp["nkill"]+df_temp["nwound"]
max_val = df_temp["casualty"].max()
display(df_temp)
plt.figure(figsize=(10,10))
plt.title("Damage done by weapons (Human)")
plt.bar(x=df_temp.index,height=df_temp["casualty"],align='center',color=df_temp["casualty"].apply(lambda x,max_val=max_val : color_from_int(x,max_val)))
plt.xticks(rotation='vertical')
plt.show()
df_temp = df_main.copy()
df_temp = df_temp[(~df_temp["weaptype1_txt"].isna() & (df_temp["property"]==1) & (~df_temp["propextent"].isna()))]
df_temp["propextent"].replace(to_replace=[1,2,3,4],value=[1e9,1e6,1e5,0],inplace=True) 
df_temp["weaptype1_txt"].replace("Vehicle [\(\)\.].*","Vehicle",regex=True,inplace=True)
df_temp = df_temp.groupby("weaptype1_txt").agg({"propextent":np.sum})
max_val = df_temp["propextent"].max()
plt.figure(figsize=(10,10))
plt.title("Damage done by weapons (property)")
plt.bar(x=df_temp.index,height=df_temp["propextent"],align='center',color=df_temp["propextent"].apply(lambda x,max_val=max_val : color_from_int(x,max_val)))
plt.xticks(rotation='vertical')
plt.show()
#People of what nationality are targeted the most?
df_temp = df_main.copy()
df_temp = df_temp[~df_temp["natlty1_txt"].isna()]
df_temp = df_temp.groupby("natlty1_txt")["eventid"].count().reset_index(name="attacks").sort_values("attacks",ascending=False)
max_val = df_temp["attacks"].max()
plt.figure(figsize=(10,10))
plt.title("Most affected nationality")
plt.bar(x=df_temp[0:15]["natlty1_txt"],height=df_temp[0:15]["attacks"],align='center',color=df_temp[0:15]["attacks"].apply(lambda x,max_val=max_val : color_from_int(x,max_val)))
plt.xticks(rotation='vertical')
plt.show()

#Various orgs analysis
#Analysis of terrorist events in India coming soon ...