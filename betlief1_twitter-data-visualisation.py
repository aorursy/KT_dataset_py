# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pysal

import plotly

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib 

import matplotlib.pyplot as plt

import plotly.plotly as py

import sklearn

import datetime

import descartes

from shapely.geometry import Point, Polygon

import plotly.graph_objs as go

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("UTF8"))

# Any results you write to the current directory are saved as output.
import geopandas as gpd



My_file_path_name = r'../input/norm2_18apl17.dbf'



Table = gpd.read_file(My_file_path_name)



#Pandas_Table = pd.DataFrame(Table)

Pandas_Table=Table

crs= {'init': 'epsg:32635'}

Pandas_Table.head()
crs= {'init': 'epsg:32635'}

geometry = [Point(xy) for xy in zip( Pandas_Table["lon"], Pandas_Table["lat"])]

geometry[:3]
import geopandas as gdp

Pandas_Table = gdp.GeoDataFrame(Pandas_Table,

                         crs=crs,

                         geometry=geometry)

Pandas_Table.head()
Pandas_Table.info()
a = Pandas_Table['tweet'].str.encode('latin-1').str.decode('utf-8', errors = 'ignore')

Pandas_Table['tweet'] = a

#datetime_object = pd.to_datetime(Pandas_Table['inserttime'])

#datetime_object.loc["2017-04-05":"2017-04-06",:]

#print(type(datetime_object))
print(type(Pandas_Table['inserttime'][1]))

datetime_object = pd.to_datetime(Pandas_Table['inserttime'])

print(type(datetime_object))

Pandas_Table["inserttime"] = datetime_object

Pandas_Table.dtypes
Pandas_Table['inserttime'] = Pandas_Table['inserttime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul')

Pandas_Table.tail()
Pandas_Table['twitteruse'].value_counts(dropna =False)  
filtered = Pandas_Table.groupby('twitteruse').filter(lambda x: len(x) <= 300)

Pandas_Table=Pandas_Table[Pandas_Table.isin(filtered)]

Pandas_Table['twitteruse'].value_counts(dropna =False)
print(type(Pandas_Table))

Pandas_Table =Pandas_Table[pd.notnull(Pandas_Table['twitteruse'])]

Pandas_Table.info()
Pandas_Table['twitteruse'].value_counts(dropna =False)  

data_fw = Pandas_Table 

data_fw['inserttime'] = pd.to_datetime(data_fw['inserttime'])

start_date = '2017-04-01 00:00:00.615668+03:00'

end_date = '2017-04-10 00:00:00.615668+03:00'

mask = (data_fw['inserttime'] > start_date) & (data_fw['inserttime'] <= end_date)

data_fw = data_fw.loc[mask]

data_fw.info()
data_sw = Pandas_Table 

data_sw['inserttime'] = pd.to_datetime(data_sw['inserttime'])

start_date = '2017-04-10 00:00:00.615668+03:00'

end_date = '2017-04-25 00:00:00.891808+03:00'

mask = (data_sw['inserttime'] > start_date) & (data_sw['inserttime'] <= end_date)

data_sw = data_sw.loc[mask]

data_sw.info()
data_wknd = Pandas_Table 

data_wknd['inserttime'] = pd.to_datetime(data_wknd['inserttime'])

start_date = '2017-04-02 01:00:09.631735+03:00'

end_date = '2017-04-03 23:59:38.891808+03:00'

mask = (data_wknd['inserttime'] > start_date) & (data_wknd['inserttime'] <= end_date)
data_wknd = data_wknd.loc[mask]

data_wknd.head()

data_wknd.info()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



stopwords = set(STOPWORDS)

stopwords.update(["https", "İstanbul", "Istanbul", "co", "Türkiye","posted","photo","Turkey"])

filtered = Table.groupby('twitteruse').filter(lambda x: len(x) >= 300)

x2011=Table[Table.isin(filtered)]

x2011 =x2011[pd.notnull(x2011['twitteruse'])]

text = " ".join(review for review in Pandas_Table.tweet)

#x2011 = Pandas_Table['twitteruse']



plt.subplots(figsize=(20,20))

wordcloud = WordCloud(    stopwords=stopwords,

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')



plt.show()
aralik=['1','2','2:4','4:6','6:8','8:10','10:300']

listemiz = [0,0,0,0,0,0,0]



#Pandas_Table['twitteruse'].value_counts(dropna =False)



a = 1

b = 1



for i in range(0,7):

    data10 = Pandas_Table

    filtered = data10.groupby('twitteruse').filter(lambda x: (len(x) <= a) & (len(x) > a-b))

    filtrelidata = data10[data10.isin(filtered)]

    listemiz[i] = filtrelidata['twitteruse'].value_counts().shape[0]

    

    if a == 1:

        a = a + 1

    elif a<=10:

        b = 2

        a = a + 2

    else:

        a = 300

        b = 290



listemiz 

barWidth = 0.25



f, ax = plt.subplots(figsize = (20,9))



# Set position of bar on X axis

r1 = np.arange(len(listemiz))

#r2 = [x + barWidth for x in r1]

 

# Make the plot

plt.bar(r1, listemiz, color='cornflowerblue', width=0.5, edgecolor='white', label='Veri')

#plt.bar(r2,liste_iki, color='crimson', width=barWidth, edgecolor='white', label='İkinci Hafta')

 

# Add xticks on the middle of the group bars

plt.xlabel('Kullanıcıların Gönderdiği Tweet Sayısı', fontweight='bold',fontsize=15)

plt.ylabel('Kullanıcı Sayısı', fontweight='bold',fontsize=15)

plt.xticks([r + 0.03 for r in range(len(listemiz))], aralik,fontsize=12)

plt.yticks(fontsize=12)

 

# Create legend & Show graphic

plt.legend(fontsize='15')

plt.show()

aralik=['1','2','2:4','4:6','6:8','8:10','10:300']

liste_ilk = [0,0,0,0,0,0,0]

liste_iki = [0,0,0,0,0,0,0]

#Pandas_Table['twitteruse'].value_counts(dropna =False)



a = 1

b = 1



for i in range(0,7):

    data5 = data_fw

    data6 = data_sw

    filtered = data5.groupby('twitteruse').filter(lambda x: (len(x) <= a) & (len(x) > a-b))

    filtrelidata = data5[data5.isin(filtered)]

    liste_ilk[i] = filtrelidata['twitteruse'].value_counts().shape[0]

    filtered = data6.groupby('twitteruse').filter(lambda x: (len(x) <= a) & (len(x) > a-b))

    filtrelidata = data6[data6.isin(filtered)]

    liste_iki[i] = filtrelidata['twitteruse'].value_counts().shape[0]

    if a == 1:

        a = a + 1

    elif a<=10:

        b = 2

        a = a + 2

    else:

        a = 300

        b = 290



liste_ilk

liste_iki



# set width of bar

barWidth = 0.25



f, ax = plt.subplots(figsize = (20,9))



# Set position of bar on X axis

r1 = np.arange(len(liste_ilk))

r2 = [x + barWidth for x in r1]

 

# Make the plot

plt.bar(r1, liste_ilk, color='cornflowerblue', width=barWidth, edgecolor='white', label='İlk Hafta')

plt.bar(r2,liste_iki, color='crimson', width=barWidth, edgecolor='white', label='İkinci Hafta')

 

# Add xticks on the middle of the group bars

plt.xlabel('Kullanıcıları Gönderdiği Tweet Sayısı', fontweight='bold')

plt.ylabel('Kullanıcı Sayısı', fontweight='bold')

plt.xticks([r + 0.12 for r in range(len(liste_ilk))], aralik)

 

# Create legend & Show graphic

plt.legend(fontsize='15')

plt.show()

df= [liste_ilk, liste_iki]

f, ax = plt.subplots(figsize = (20,9))

sns.barplot(x=liste_ilk,y=aralik,color='orange',alpha = 0.5,label='First Week' )

sns.barplot(x=liste_iki,y=aralik,color='red',alpha = 0.5,label='Second Week')





ax.legend(loc='lower right',frameon = True,fontsize='15')     # legendlarin gorunurlugu

ax.set(xlabel='Tweet Sayısı', ylabel='Tweet Sayısı Aralıkları',title = "Haftalara göre tweet")
data_sw['inserttime'] = pd.to_datetime(data_sw['inserttime'])

start_date = '2017-04-11 18:11:39.615668+03:00'

end_date = '2017-04-25 19:26:38.891808+03:00'

mask = (data_sw['inserttime'] > start_date) & (data_sw['inserttime'] <= end_date)
x = ['Pzts','Salı','Çarş','Perş','Cuma','Cmrts','Pzr']

y = [0,0,0,0,0,0,0]



start_date = pd.to_datetime('2017-04-10 00:00:00.000000').tz_localize('Europe/Istanbul')

end_date =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')

for i in range(0,7):

    start_date +=  pd.Timedelta('1 days')

    end_date +=  pd.Timedelta('1 days')

    mask = (data_sw['inserttime'] > start_date) & (data_sw['inserttime'] <= end_date)

    y[i] = data_sw.loc[mask].shape[0]

y



#visualization

plt.figure(figsize=(10,10))

sns.barplot(x=x, y=y,palette= sns.diverging_palette(255, 133, l=60, n=7, center="dark"))

plt.xticks(rotation= 90)

plt.ylabel('Total Tweet',fontsize = 15)

plt.grid()
x = ['Pzts','Salı','Çarş','Perş','Cuma','Cmrts','Pzr']

y = [0,0,0,0,0,0,0]



start_date = pd.to_datetime('2017-04-10 00:00:00.000000').tz_localize('Europe/Istanbul')

end_date =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')

for i in range(0,7):

    start_date +=  pd.Timedelta('1 days')

    end_date +=  pd.Timedelta('1 days')

    mask = (data_sw['inserttime'] > start_date) & (data_sw['inserttime'] <= end_date)

    u = data_sw.loc[mask]

    y[i]=len(u['twitteruse'].value_counts(dropna =False))



#visualization

plt.figure(figsize=(10,10))

sns.barplot(x=x, y=y,color='grey',alpha=0.5)

plt.xticks(rotation= 90)

plt.ylabel('Kişi Sayısı',fontsize = 15)

plt.grid()
x = [0,1, 2,3, 4,5, 6,7, 8,9, 10,11,12,13, 14,15, 16,17, 18,19, 20,21, 22,23]

y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

c = ['b','g','r','c','y','k','cornflowerblue']



start_date = pd.to_datetime('2017-04-10 00:00:00.000000').tz_localize('Europe/Istanbul')

end_date =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')

start_time = pd.to_datetime('2017-04-10 22:00:00.000000').tz_localize('Europe/Istanbul')

end_time =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')



f,ax1 = plt.subplots(figsize =(20,10))



for i in range(0,7):

    start_date +=  pd.Timedelta('1 days')

    end_date +=  pd.Timedelta('1 days')

    mask = (data_sw['inserttime'] >= start_date) & (data_sw['inserttime'] <= end_date)

    elbetbirgun = data_sw.loc[mask]

    for j in range(0,24):

        start_time +=  pd.Timedelta('1 hours')

        end_time +=  pd.Timedelta('1 hours')

        mask = (elbetbirgun['inserttime'] >= start_time) & (elbetbirgun['inserttime'] <= end_time)

        y[j] = elbetbirgun.loc[mask].shape[0]

    sns.pointplot(x=x,y=y,color=c[i],alpha=0.8, linestyles='--')

    plt.text(0,4000-(i*200),str(i+1) + '. Gün',color=c[i],fontsize = 17,style = 'italic')



plt.xlabel('Saatler',fontsize = 15,color='blue')

plt.ylabel('Tweet Sayısı',fontsize = 15,color='blue')

plt.grid()

 

x = [0,1, 2,3, 4,5, 6,7, 8,9, 10,11,12,13, 14,15, 16,17, 18,19, 20,21, 22,23]

y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

c = ['b','g','r','c','y','k','cornflowerblue']



start_date = pd.to_datetime('2017-04-10 00:00:00.000000').tz_localize('Europe/Istanbul')

end_date =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')

start_time = pd.to_datetime('2017-04-10 22:00:00.000000').tz_localize('Europe/Istanbul')

end_time =  pd.to_datetime('2017-04-11 00:00:00.000000').tz_localize('Europe/Istanbul')



f,ax1 = plt.subplots(figsize =(20,10))



for i in range(0,7):

    start_date +=  pd.Timedelta('1 days')

    end_date +=  pd.Timedelta('1 days')

    mask = (data_sw['inserttime'] >= start_date) & (data_sw['inserttime'] <= end_date)

    elbetbirgun = data_sw.loc[mask]

    for j in range(0,24):

        start_time +=  pd.Timedelta('1 hours')

        end_time +=  pd.Timedelta('1 hours')

        mask = (elbetbirgun['inserttime'] >= start_time) & (elbetbirgun['inserttime'] <= end_time)

        un = elbetbirgun.loc[mask]

        y[j]=len(un['twitteruse'].value_counts(dropna =False))

    sns.pointplot(x=x,y=y,color=c[i],alpha=0.8, linestyles='--')

    plt.text(0,2900-(i*150),str(i+1) + '. Gün',color=c[i],fontsize = 15,style = 'italic')



plt.xlabel('Saatler',fontsize = 15,color='blue')

plt.ylabel('Kullanıcı Sayısı',fontsize = 15,color='blue')

plt.grid()

 

    
date = pd.to_datetime('2017-04-11 18:11:39.615668+03:00')

for i in range(5): 

    date += pd.Timedelta('1 days')

    print(date)

    



import folium 



icon = folium.features.CustomIcon('https://upload.wikimedia.org/wikipedia/commons/1/19/Twitter_icon.svg',icon_size=(14, 14))



base_m = folium.Map(location=[40.909550, 29.389620], control_scale=True, zoom_start=10)

for lat,lng,num in zip(data_wknd.lat,data_wknd.lon,range(1,data_wknd.shape[0])): 

        #popup = folium.Popup(data_wknd['twitteruse'][num], parse_html=True)

        #folium.Marker( location=[ lat, lng ]).add_to(base_map)

        if num<300:

            folium.Marker( location=[ lat, lng ], popup=data_wknd.iloc[num]['tweet']).add_to(base_m)

    

base_m
fig, ax = plt.subplots(figsize=(15,15))

Table.plot(ax=ax, alpha= 0.4, color="grey")

Pandas_Table[Pandas_Table['lat']>40].plot(ax=ax, markersize=20, color="blue", marker = "o", label="Neg")

Pandas_Table[Pandas_Table['lon']>29.0458].plot(ax=ax, markersize=20, color="red", marker = "^", label="Pos")

plt.legend(prop={'size':15})

import folium 

def generateBaseMap(default_location=[41.109550, 28.989620], default_zoom_start=10):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map

base_map = generateBaseMap()

base_map
from folium.plugins import HeatMap

df_copy = data_wknd.copy()

df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=df_copy[['lat', 'lon', 'count']].groupby(['lat', 'lon']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)
df_copy.head()
base_map
arrayunique=data_wknd.twitteruse.unique()





yenidf = pd.DataFrame(arrayunique, columns =['twitteruse']) 

yenidf['lat']=0

yenidf['lon']=0

yenidf['tweet']=""





def fonkfonk(ts):

    yuyo=data_wknd[data_wknd["twitteruse"] == ts]

    return yuyo.sum(axis = 0, skipna = True).lat/yuyo['lat'].shape[0]



def fonkfonk2(ts):

    yuyo=data_wknd[data_wknd["twitteruse"] == ts]

    return yuyo.sum(axis = 0, skipna = True).lon/yuyo['lon'].shape[0]



def fonkfonk3(ts):

    yuyo=data_wknd[data_wknd["twitteruse"] == ts]

    return yuyo.sum(axis = 0, skipna = True).tweet



yenidf['lat'] = yenidf['twitteruse'].apply(fonkfonk)

yenidf['lon'] = yenidf['twitteruse'].apply(fonkfonk2)

yenidf['tweet'] = yenidf['twitteruse'].apply(fonkfonk3)



yenidf
import folium 



icon = folium.features.CustomIcon('https://upload.wikimedia.org/wikipedia/commons/1/19/Twitter_icon.svg',icon_size=(14, 14))



base_m = folium.Map(location=[40.909550, 29.389620], control_scale=True, zoom_start=10)

for lat,lng,num in zip(yenidf.lat,yenidf.lon,range(1,yenidf.shape[0])): 

        #popup = folium.Popup(data_wknd['twitteruse'][num], parse_html=True)

        #folium.Marker( location=[ lat, lng ]).add_to(base_map)

        if(num<3000):

            folium.Marker( location=[ lat, lng ]).add_to(base_m)

    

base_m
data_gun1 = Pandas_Table 

data_gun1['inserttime'] = pd.to_datetime(data_gun1['inserttime'])

start_date = '2017-04-10 00:00:00.615668+03:00'

end_date = '2017-04-11 00:00.891808+03:00'

mask = (data_gun1['inserttime'] > start_date) & (data_gun1['inserttime'] <= end_date)

data_gun1 = data_gun1.loc[mask]

data_gun1.info()
data_gun1['hour'] = 0



def hr_func(ts):

    return ts.hour



data_gun1['hour'] = data_gun1['inserttime'].apply(hr_func)

df_hour_list = []

gun = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

for hour in gun:

    df_hour_list.append(data_gun1.loc[data_gun1.hour == hour, ['lat', 'lon']].groupby(['lat', 'lon']).sum().reset_index().values.tolist())
from folium.plugins import HeatMapWithTime

base_map3 = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_hour_list, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},

                min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map3)

base_map3
gunluk = data_fw

gunluk['day'] = 0



def hr_func_2(ts):

    return ts.day



gunluk['day'] = gunluk['inserttime'].apply(hr_func_2)

df_day_list = []

gun = [1,2,3,4,5,6,7,8,9,10,11]

for gun in gun:

    df_day_list.append(gunluk.loc[gunluk.day == gun, ['lat', 'lon']].groupby(['lat', 'lon']).sum().reset_index().values.tolist())

from folium.plugins import HeatMapWithTime

base_map4 = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_day_list, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},

                min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map4)

base_map4
