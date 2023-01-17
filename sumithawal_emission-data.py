# pip install opencage
from matplotlib import style 
style.use('fivethirtyeight')
from opencage.geocoder import OpenCageGeocode
from mpl_toolkits.basemap import Basemap
geocoder = OpenCageGeocode(key)
key = '2117b047581540e59c772059f03a714a'
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from wordcloud import WordCloud 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/co2-ghg-emissionsdata/co2_emission.csv')

df.head()
df.describe()
# top 5 least emission zones 
def bar_graph_1(df):
    fig = plt.figure(figsize=(10,6))
    y = df.groupby(['Entity']).sum().sort_values('Annual CO₂ emissions (tonnes )')['Annual CO₂ emissions (tonnes )'][1:6]
    x_labels =[y.index[i] for i in range(len(y))]
    x =[1,2,3,4,5]
    plt.bar(x,y)
    plt.xticks(x,x_labels,fontsize=12)
    plt.xlabel('Countries',fontsize=12)
    plt.ylabel('Emission',fontsize=12)
    plt.show()
bar_graph_1(df)    
# top 5 emissions 
def bar_graph_2(df):
    
    fig = plt.figure(figsize=(10,6))
    y = df.groupby(['Entity']).sum().sort_values('Annual CO₂ emissions (tonnes )')['Annual CO₂ emissions (tonnes )'][-5:]
    x_labels =[y.index[i] for i in range(len(y))]
    x =[1,2,3,4,5]
    plt.bar(x,y)
    plt.xticks(x,x_labels,fontsize=12)
    plt.xlabel('Countries',fontsize=12)
    plt.ylabel('Emission',fontsize=12)
    plt.show()
bar_graph_2(df)
# top 20 countries emission
def time_series_plot1(df):
    style.use('ggplot')
    fig = plt.figure(figsize=(12,6))
    y = df.groupby(['Entity']).sum().sort_values('Annual CO₂ emissions (tonnes )')['Annual CO₂ emissions (tonnes )'][-20:]
    x_labels =[y.index[i] for i in range(len(y))]
    for i in range(len(x_labels)):       
        abc=df.groupby(['Entity','Year']).mean().loc[x_labels[i]]
        plt.plot(abc.index,abc['Annual CO₂ emissions (tonnes )'])
    plt.xlabel('Years',fontsize=20)
    plt.ylabel('Emission_over_time',fontsize=20)
    plt.legend(x_labels)
time_series_plot1(df)
#least 20 countries emission
def time_series_plot2(df):
    style.use('seaborn')
    fig = plt.figure(figsize=(12,6))
    y = df.groupby(['Entity']).sum().sort_values('Annual CO₂ emissions (tonnes )')['Annual CO₂ emissions (tonnes )'][1:20]
    x_labels =[y.index[i] for i in range(len(y))]
    for i in range(len(x_labels)):
        abc=df.groupby(['Entity','Year']).mean().loc[x_labels[i]]
        plt.plot(abc.index,abc['Annual CO₂ emissions (tonnes )'])
    plt.xlabel('Years',fontsize=20)
    plt.ylabel('Emission_over_time',fontsize=20)
    plt.legend(x_labels)
time_series_plot2(df)
def word_cloud1(df):
    x = df['Entity'].value_counts()
    word_ls =[]
    for i in range(len(x)):
        word_ls += [x.index[i]]*x[i]
   
    wordcloud = WordCloud(width=350,height=250).generate(''.join(word_ls))
    plt.figure(figsize=(19,9))
    plt.axis('off')
    plt.title('Countries',fontsize=20)
    plt.imshow(wordcloud)
    plt.show()
word_cloud1(df)
def only_country_names():
    new_list =[]
    ls = df.groupby(['Entity']).mean().sort_values(['Annual CO₂ emissions (tonnes )'])[-51:-1].index
    for i in range(len(ls)):
        new_list.append(ls[i].split(' ')[0])
    return new_list
ls = only_country_names()

# top 50 countries emission plotting 
def plot_world_basemap():
    fig = plt.figure(figsize=(14,20))
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()
    m.drawstates()

    
    for i in range(len(ls)):
        query = ls[i]
        results = geocoder.geocode(query)
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        x,y = m(lng,lat)
        m.plot(x,y,'o','red',markersize=(i/2)*2)
    
    m.fillcontinents(color='white',lake_color='#FFFFFF')

    m.drawmapboundary(fill_color='#FFFFFF')
    plt.title("Emissions by top 50 countries",fontsize=20)
    plt.show()


plot_world_basemap()


def hist_plot(df):
    y =[]
    x_labels =[]
    for i in range(18):
    
        x=df[df['Year']==2000+i].sort_values(['Annual CO₂ emissions (tonnes )','Year'])
        intm = x['Annual CO₂ emissions (tonnes )'][-2:-1].values
        label =x['Year'][-2:-1].values
        x_labels.append(label[0])
        y.append(int(intm[0]))
      
    x =[x for x in range(18)]
    return (x,y,x_labels)
fig= plt.figure(figsize=(14,8))

x,y,x_labels =hist_plot(df)
plt.bar(x[:6],y[:6],color='blue')
plt.bar(x[6:],y[6:],color='red')
plt.xticks(x,x_labels,rotation=45)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Most emmissions by single country',fontsize=20)
plt.legend(['USA','China'])

def pie_chart1(df):
    fig,axes =  plt.subplots(3,2,figsize=(12,10))
    plt.title('top 5 polluting regions every 50 years')
    x = df[df['Year']==1751]
    x = x.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x['Entity'][:5]
    axes[0,0].pie(values,labels=labels,startangle=45,explode=[0.0,0.0,0.0,0.1,0.3])
    axes[0,0].set_title('1751',fontsize=15)
    
    x1 = df[df['Year']==1800]
    x1 = x1.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x1['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x1['Entity'][:5]
    axes[0,1].pie(values,labels=labels,startangle=45,explode=[0.0,0.0,0.0,0.1,0.3])
    axes[0,1].set_title('1800',fontsize=15)
    
    
    x2 = df[df['Year']==1850]
    x2 = x2.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x2['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x2['Entity'][:5]
    axes[1,0].pie(values,labels=labels,startangle=45)
    axes[1,0].set_title('1850',fontsize=15)
    
    
    x3 = df[df['Year']==1900]
    x3 = x3.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x3['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x3['Entity'][:5]
    axes[1,1].pie(values,labels=labels)
    axes[1,1].set_title('1900',fontsize=15)
    
    x4 = df[df['Year']==1950]
    x4 = x4.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x4['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x4['Entity'][:5]
    axes[2,0].pie(values,labels=labels)
    axes[2,0].set_title('1950',fontsize=15)
    
    x5 = df[df['Year']==2000]
    x5 = x5.sort_values('Annual CO₂ emissions (tonnes )',ascending=False)
    values = x5['Annual CO₂ emissions (tonnes )'][:5]
    labels=  x5['Entity'][:5]
    axes[2,1].pie(values,labels=labels)
    axes[2,1].set_title('2000',fontsize=15)
    
    plt.show()
pie_chart1(df)