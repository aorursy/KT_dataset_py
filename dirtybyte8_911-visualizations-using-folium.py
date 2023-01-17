import geopandas as gpd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import folium

import datetime as dt

font={

    'size':20

}

sns.set(style="white", color_codes=True)
fp = "../input/911.csv"
dateparse = lambda x: pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

df= pd.read_csv(fp,parse_dates=['timeStamp'],date_parser=dateparse)
df.head()
df.dtypes
df.set_index('timeStamp',inplace=True)

df.head(3)
df.dtypes
df['type'] = df["title"].apply(lambda x: x.split(':')[0].strip())
df.head(3)
print("The frequency of emergency type is:\n",df["type"].value_counts())
plt.figure(figsize=(12,8))

sns.countplot(x=df['type'],data=df,palette='Spectral')

plt.xlabel('Emergency Types',fontdict=font)

plt.ylabel('counts',fontdict=font)

plt.title("Total number of calls",fontdict=font) 

plt.savefig('Total-calls.png')

plt.show()

tab=pd.crosstab(df['twp'],df['type']) 
tab.head(10)
ems = pd.DataFrame(tab[['EMS']])

ems.sort_values(by='EMS',axis=0, ascending=False, inplace=True)
ems.dtypes
ems.head()
plt.figure(figsize=(12,8))

temp=ems[['EMS']].iloc[:10,:]

x_list = temp['EMS']

label_list = temp.index

plt.axis("equal") 

#To show the percentage of each pie slice, pass an output format to the autopctparameter 

plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 

plt.title("EMS calls recieved from top 10 cities",fontdict=font) 

plt.savefig('ems-pie-Top10.png')

plt.show()

fire = pd.DataFrame(tab[['Fire']])

fire.sort_values(by='Fire',axis=0, ascending=False, inplace=True)
fire.head()
plt.figure(figsize=(12,8))

temp=fire[['Fire']].iloc[:10,:]

x_list = temp['Fire']

label_list = temp.index

plt.axis("equal") 

#To show the percentage of each pie slice, pass an output format to the autopctparameter 

plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 

plt.title("Fire calls recieved from top 10 cities",fontdict=font) 

plt.savefig('fire-pie-Top10.png')

plt.show()
traffic = pd.DataFrame(tab[['Traffic']])

traffic.sort_values(by='Traffic',axis=0, ascending=False, inplace=True)
traffic.head()
plt.figure(figsize=(12,8))

temp=traffic[['Traffic']].iloc[:10,:]

x_list = temp['Traffic']

label_list = temp.index

plt.axis("equal") 

#To show the percentage of each pie slice, pass an output format to the autopctparameter 

plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 

plt.title("Traffic calls recieved from top 10 cities",fontdict=font) 

plt.savefig('Traffic-pie-Top10.png')

plt.show()

ems = df[df['type']=='EMS']

ems['type'] = ems["title"].apply(lambda x: x.split(':')[1].strip())

ems.head()
ems_type = pd.DataFrame(ems['type'].value_counts(sort=True, ascending=False)).iloc[:10,:]

ems_type.head()
plt.figure(figsize=(12,8))

sns.barplot(x=ems_type['type'],y=ems_type.index,data=ems_type,palette="viridis") 

plt.xlabel("Count",fontdict=font)

plt.ylabel("Type of EMS Emergency",fontdict=font)

plt.title("Frequently appeared ( top 10 ) type of EMS Emergency",fontdict=font)

plt.savefig('Top10-EMS-Emergency.png')

plt.show()

fire = df[df['type']=='Fire']

fire['type'] = fire["title"].apply(lambda x: x.split(':')[1].strip())

fire.head()
fire_type = pd.DataFrame(fire['type'].value_counts(sort=True, ascending=False)).iloc[:10,:]

fire_type.head()
plt.figure(figsize=(12,8))

sns.barplot(x=fire_type['type'],y=fire_type.index,data=fire_type,palette="viridis") 

plt.xlabel("Count",fontdict=font)

plt.ylabel("Type of Fire Emergency",fontdict=font)

plt.title("Frequently appeared ( top 10 ) type of Fire Emergency",fontdict=font)

plt.savefig('Top10-Fire-Emergency.png')

plt.show()

traffic = df[df['type']=='Traffic']

traffic['type'] = traffic["title"].apply(lambda x: x.split(':')[1].strip())

traffic.head()
traffic_type = pd.DataFrame(traffic['type'].value_counts(sort=True, ascending=False)).iloc[:10,:]

traffic_type.head()
plt.figure(figsize=(12,8))

sns.barplot(x=traffic_type['type'],y=traffic_type.index,data=traffic_type,palette="viridis") 

plt.xlabel("Count",fontdict=font)

plt.ylabel("Type of Traffic Emergency",fontdict=font)

plt.title("Frequently appeared ( top 10 ) type of Traffic Emergency",fontsize=18)

plt.savefig('Top10-Traffic-Emergency.png')

plt.show()

df.head()
tempdate = df.index

df['tempdate']=tempdate

df.head(3)
df['year'] = df['tempdate'].dt.year

df.head(3)
df['month'] = df['tempdate'].dt.month_name()

df.head(3)
df['day']=df['tempdate'].dt.day_name()

df.head(3)
df['hours'] =df['tempdate'].dt.hour

df.head(3)
df.drop(['tempdate'],axis=1,inplace=True)
df.head(3)
calls_month = df.groupby(['month', 'type'])['type'].count()
calls_month

calls_month_percent = calls_month.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
calls_month_percent
month_seq = [dt.date(2019, m, 1).strftime('%B') for m in range(1, 13)]
month_seq
#reindexing level 0

calls_month_percent=calls_month_percent.reindex(month_seq, level=0)
calls_month_percent
# reindexing level 1

calls_month_percent = calls_month_percent.reindex(['EMS','Traffic','Fire'], level=1)
calls_month_percent
sns.set(rc={'figure.figsize':(12, 8)})

calls_month_percent.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=30)

plt.title('Calls/Month', fontdict=font)

plt.savefig('call_vs_month.png')
calls_day = df.groupby(['day','type'])['type'].count()
calls_day
calls_day_percent = calls_day.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
calls_day_percent
import calendar

day_seq = list(calendar.day_name)

day_seq
calls_day_percent=calls_day_percent.reindex(day_seq, level=0)

calls_day_percent = calls_day_percent.reindex(['EMS','Traffic','Fire'], level=1)
calls_day_percent
sns.set(rc={'figure.figsize':(12, 8)})

calls_day_percent.unstack().plot(kind='bar')

plt.xlabel('Name of the Day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=30)

plt.title('Calls/Day', fontdict=font)

plt.savefig('call_vs_day.png')
calls_hour = df.groupby(['hours','type'])['type'].count()
calls_hour
calls_hour_percent = calls_hour.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
calls_hour_percent
calls_hour_percent = calls_hour_percent.reindex(['EMS','Traffic','Fire'], level=1)
calls_hour_percent
sns.set(rc={'figure.figsize':(12, 8)})

calls_hour_percent.unstack().plot(kind='bar')

plt.xlabel('Hour', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=30)

plt.title('Calls/Hour', fontdict=font)

plt.savefig('call_vs_hour.png')
# Extracting data for EMS

ems_city = df[df['type']=='EMS']
ems_city.shape
ems_city.head()
# Extracting data for Traffic

traffic_city =df[df['type']=='Traffic']
traffic_city.shape
traffic_city.head()
# Extracting Data for Fire

fire_city = df[df['type']=='Fire']
fire_city.shape
fire_city.head()
gdf_traffic = gpd.GeoDataFrame(traffic_city)

gdf_traffic.head()
gdf_traffic.dtypes
new_geo = gpd.GeoDataFrame()

new_geo['lat']=gdf_traffic['lat'].astype('float64')

new_geo['lng']=gdf_traffic['lng'].astype('float64')

new_geo['title']=gdf_traffic['title'].astype('object')

location = new_geo['lat'].mean(), new_geo['lng'].mean()



new_geo.shape
new_geo.head()

locationlist = new_geo[['lat','lng']].values.tolist()

labels = "CITY => "+ gdf_traffic['twp']+"\n"+gdf_traffic['title']



m = folium.Map(location=location, zoom_start=10)





for point in range(1,100): 

    popup = folium.Popup(labels[point], parse_html=True)

    icon = folium.Icon(color='orange')

    folium.Marker(locationlist[point], popup=popup, icon=icon).add_to(m)

    

m.save(outfile= "Traffic.html")

m
gdf_fire = gpd.GeoDataFrame(fire_city)

gdf_fire.head()
new_geo = gpd.GeoDataFrame()

new_geo['lat']=gdf_fire['lat'].astype('float64')

new_geo['lng']=gdf_fire['lng'].astype('float64')

location = new_geo['lat'].mean(), new_geo['lng'].mean()



new_geo.shape
new_geo.head()
locationlist = new_geo[['lat','lng']].values.tolist()



labels = "CITY => "+ gdf_fire['twp']+"\n"+gdf_fire['title']



m = folium.Map(location=location, zoom_start=10)



for point in range(1,100): 

    popup = folium.Popup(labels[point], parse_html=True)

    icon = folium.Icon(color='purple')

    folium.Marker(locationlist[point], popup=popup,icon=icon).add_to(m)



m.save(outfile="Fire.html")

m
gdf_ems = gpd.GeoDataFrame(ems_city)

gdf_ems.head()
new_geo = gpd.GeoDataFrame()

new_geo['lat']=gdf_ems['lat'].astype('float64')

new_geo['lng']=gdf_ems['lng'].astype('float64')

location = new_geo['lat'].mean(), new_geo['lng'].mean()



new_geo.shape
new_geo.head()
locationlist = new_geo[['lat','lng']].values.tolist()



labels = "CITY => "+ gdf_ems['twp']+"\n"+gdf_ems['title']



m = folium.Map(location=location, zoom_start=10)



for point in range(1,100): 

    popup = folium.Popup(labels[point], parse_html=True)

    icon = folium.Icon(color='blue')

    folium.Marker(locationlist[point], popup=popup,icon=icon).add_to(m)

    

m.save(outfile= "EMS.html")

m
gdf = gpd.GeoDataFrame(df)

gdf.head()
new_geo = gpd.GeoDataFrame()

new_geo['lat']=gdf['lat'].astype('float64')

new_geo['lng']=gdf['lng'].astype('float64')

new_geo['type']=gdf['type'].astype('object')

location = new_geo['lat'].mean(), new_geo['lng'].mean()



new_geo.shape
locationlist = new_geo[['lat','lng']].values.tolist()

labels = "CITY => "+ gdf['twp']+"\n"+gdf['title']

etype = gdf['type'].values.tolist()



m = folium.Map(location=location, zoom_start=10)



for point in range(1,300):

    if(etype[point] == 'EMS'):

        icon = folium.Icon(color='blue')

        label=labels[point]

    elif(etype[point]=='Traffic'):

        icon = folium.Icon(color='purple')

        label=labels[point]

    elif(etype[point]=='Fire'):

        icon = folium.Icon(color='red')

        label=labels[point]

    

    popup = folium.Popup(label, parse_html=True)

    folium.Marker(locationlist[point], popup=popup, icon=icon).add_to(m)

    

m.save(outfile= "All-Map-In-One.html")

m