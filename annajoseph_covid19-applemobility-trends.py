import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Reading the dataset into pandas
df=pd.read_csv("/kaggle/input/uncover/UNCOVER/apple_mobility_trends/mobility-trends.csv")
df.head()
df.describe()
df.dropna(subset=['date'],axis='rows',inplace=True)
df['pdate']=pd.to_datetime(df.date,format="%Y-%m-%d")
df["dayofweek"]=df.pdate.dt.dayofweek
#df.date=df.date.dt.strftime("%Y-%m-%d")
gr_region=df.groupby("region")
gr_date=df.groupby("date")
df_region=gr_region.sum()
df_date=gr_date.sum()
df_india=df[df.region=='India']


df_date.head()
df.head()
#df_india.transportation_type.unique()
plt.figure(figsize=(24,10))
plot1=sns.lineplot(x=df_date.index,y='value',data=df_date)
plot1.set_xticklabels(df_date.index,rotation=90)
plot1.set_title("Trend of Direction request")
plt.figure(figsize=(24,10))
plot2=sns.lineplot(x='date',y='value',data=df_india[df['transportation_type']=='driving'],legend='brief',label='Driving')
sns.lineplot(x='date',y='value',data=df_india[df['transportation_type']=='walking'],legend='brief',label='Walking')
plot2.set_xticklabels(df_india[df_india['transportation_type']=="driving"].date,rotation=90)
plot2.set_title("Trend of Direction request in India",fontsize=30)
plot2.set_xlabel('Date',fontsize=20)
plot2.set_ylabel('Relative Volume of Direction Requests',fontsize=20)
df_india.dtypes
df.dayofweek
df.region.unique().size
df[df['date']=='2020-01-13']
plt.figure(figsize=(24,20))
plot3=sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date<='2020-03-23'],legend='brief',label='Before Lockdown')
sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date>'2020-03-23'],legend='brief',label='During Lockdown')
xlabels={'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'}
plot3.set_xticklabels(xlabels,rotation=90)
plot3.set_title("Driection Request Trend against Day of Week",fontsize=30)
plot3.set_xlabel('Day of Week',fontsize=20)
plot3.set_ylabel('Relative Volume of Direction Requests',fontsize=20)

#sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date>'2020-03-22'],legend='brief',label='During Lock down')
#A function to mark a date as ALS(After Lockdown Start, or BLS - before lockdown start) for engineering categorical variable.
def markPeriod(dt):
    if(dt>'2020-03-23'):
        return "ALS" #After Lockdown Start
    else:
        return "BLS"  #Before Lockdown Start
#Creating a categorical variable
df_india['period']=df_india.apply(lambda x : markPeriod(x['date']),axis=1)
plt.figure(figsize=(24,20))
plot4=sns.boxplot(x='transportation_type',y='value',data=df_india[df_india.period=='ALS'], notch=True)
plot4.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot4.set_title("Boxplot of Transportation Type",fontsize=30)
plot4.set_xlabel('Transportation Type',fontsize=20)
plot4.set_ylabel('Relative Volume of Direction Requests',fontsize=20)

plt.figure(figsize=(24,20))
plot5=sns.boxplot(x='period',y='value',data=df_india, notch=True)
plot5.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot5.set_title("Boxplot of Time Period",fontsize=30)
plot5.set_xlabel('Before/After Lockdown',fontsize=20)
plot5.set_ylabel('Relative Volume of Direction Requests',fontsize=20)
plt.figure(figsize=(24,20))
sns.distplot(df[df['region']=='India'].value)
plot5.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot5.set_title("Boxplot of Time Period",fontsize=30)
plot5.set_xlabel('Before/After Lockdown',fontsize=20)
plot5.set_ylabel('Relative Volume of Direction Requests',fontsize=20)
import folium
from folium import plugins
import json
import os
json_path = os.path.join(os.getcwd(),'mydataset/','countries.geojson') 
world_geo = json.load(open("/kaggle/input/my-input-data/countries.geojson"))
#Code to try assign countries for cities, and subregions.
#code takes lots of time to run, need to try for alternative machanism.
df_citymap=pd.read_csv("/kaggle/input/my-input-data/world-cities_csv.csv")
def getCountry(city):
    #print(city)
    country=df_citymap[df_citymap.country==city].country.unique()
    if(country.size!=0):
        return country[0]
    country=df_citymap[df_citymap.subcountry==city].country.unique()
    if(country.size!=0):
         return country[0]
    country = df_citymap[df_citymap.name==city].country
    return "".join(country.tail(1))
    return city

#print(getCountry('Chennai'))
#df_temp=df.copy()
#df_temp['country']=df_temp.apply(lambda x: getCountry(x['region']),axis=1)
#df_temp.head()

df['period']=df.apply(lambda x : markPeriod(x['date']),axis=1)
df_countrywise_BLS=df[df.period=='BLS']
df_countrywise_ALS=df[df.period=='ALS']
df_countrywise_BLS=df_countrywise_BLS[(df_countrywise_BLS.geo_type=='country/region')|(df_countrywise_BLS.geo_type=='sub-region')].groupby('region').sum()
df_countrywise_ALS=df_countrywise_ALS[(df_countrywise_ALS.geo_type=='country/region')|(df_countrywise_ALS.geo_type=='sub-region')].groupby('region').sum()
df_countrywise_BLS.head()
#Below is a word map overlayed with data about cumulative Direction requests in intial days of COVID.

world_map_BLS = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map_BLS.choropleth(
    geo_data=world_geo,
    data=df_countrywise_BLS,
    columns=[df_countrywise_BLS.index, 'value'],
    key_on='feature.properties.ADMIN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Direction Requests'
)
world_map_BLS

#Below is a word map overlayed with data about cumulative Direction requests of recent past.

world_map_ALS = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map_ALS.choropleth(
    geo_data=world_geo,
    data=df_countrywise_ALS,
    columns=[df_countrywise_ALS.index, 'value'],
    key_on='feature.properties.ADMIN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Direction Requests'
)
world_map_ALS
