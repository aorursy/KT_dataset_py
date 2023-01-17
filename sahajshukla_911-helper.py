import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import fiona
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/montcoalert/911.csv")
print(df.describe())
print("the columns are: \n ",df.columns)
print("Sample Data: \n", df.head())
df1 = df[df["twp"]=="LOWER POTTSGROVE"]
df1
for i in df.iloc[:,6]:
    if (i=="LOWER POTTSGROVE"):
        df["zip"] = 19464.0
df['zip'].isna().sum()
df = df[(df['lng']>=-75.7) & (df['lng']<=-75.0)]
df = df[(df['lat']>=39.8) & (df['lat']<=40.5)]
print(df['lat'].max())
print(df['lat'].min())
print(df['lng'].max())
print(df['lng'].min())
df['title'].nunique()
df['Reason']=df['title'].apply(lambda x:x.split(':')[0])
df['Reason'].unique()
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Reason'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Reason for Call')
ax[0].set_ylabel('Count')
sns.countplot('Reason',data=df,ax=ax[1],order=df['Reason'].value_counts().index)
ax[1].set_title('Count of Reason')
plt.show()
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['Hour']=df['timeStamp'].apply(lambda x:x.hour)
df['Month']=df['timeStamp'].apply(lambda x:x.month)
df['DayOfWeek']=df['timeStamp'].apply(lambda x:x.dayofweek)
byMonth=df.groupby('Month').count()
byMonth['lat'].plot();
plt.title("line graph of 911 calls distribution per month")
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index());
plt.title("linear Model of 911 calls distribution per month")
df.head()
df_1 = df[df['Reason']=="EMS"]
df_1['timeStamp']=pd.to_datetime(df_1['timeStamp'])
df_1['Hour']=df_1['timeStamp'].apply(lambda x:x.hour)
df_1['Month']=df_1['timeStamp'].apply(lambda x:x.month)
df_1['DayOfWeek']=df_1['timeStamp'].apply(lambda x:x.dayofweek)
byMonth_1=df_1.groupby('Month').count()
byMonth_1['lat'].plot();
plt.title("line graph of EMS calls distribution per month")
sns.lmplot(x='Month',y='twp',data=byMonth_1.reset_index());
plt.title("linear Model of EMS calls distribution per month")
df_2 = df[df['Reason']=="Fire"]
df_2['timeStamp']=pd.to_datetime(df_2['timeStamp'])
df_2['Hour']=df_2['timeStamp'].apply(lambda x:x.hour)
df_2['Month']=df_2['timeStamp'].apply(lambda x:x.month)
df_2['DayOfWeek']=df_2['timeStamp'].apply(lambda x:x.dayofweek)
byMonth_2=df_2.groupby('Month').count()
byMonth_2['lat'].plot();
plt.title("line graph of Fire calls distribution per month")
sns.lmplot(x='Month',y='twp',data=byMonth_2.reset_index());
plt.title("linear Model of Fire calls distribution per month")
df_3 = df[df['Reason']=='Traffic']
df_3['timeStamp']=pd.to_datetime(df_3['timeStamp'])
df_3['Hour']=df_3['timeStamp'].apply(lambda x:x.hour)
df_3['Month']=df_3['timeStamp'].apply(lambda x:x.month)
df_3['DayOfWeek']=df_3['timeStamp'].apply(lambda x:x.dayofweek)
byMonth_3=df.groupby('Month').count()
byMonth_3['lat'].plot();
plt.title("line graph of Traffic calls distribution per month")
sns.lmplot(x='Month',y='twp',data=byMonth_3.reset_index());
plt.title("linear Model of Traffic calls distribution per month")
street_map = gpd.read_file(r"../input/map-files/tl_2018_42091_roads.shp")
df.drop(['title'], axis = 1, inplace= True)
df.head()
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax)
df_new = df
crs = {'init':'EPSG:4326'}
#setting our coordinate system
geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]
geometry[:3]
geo_df = gpd.GeoDataFrame(df,crs = crs, geometry = geometry)
geo_df.drop(['lat','lng', 'desc', 'addr', 'e', 'timeStamp', 'zip', 'twp'], axis = 1, inplace = True)
geo_df.head()
geo_df = geo_df.iloc[5000:10000,:]
# Randomly taking 5000 entries to map. Looks very untidy otherwise
len(geo_df)
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax,alpha = 0.4, color = "grey")
geo_df[geo_df['Reason']=='Fire'].plot(ax = ax, markersize=20, color = "orange", marker = "*",label = "Fire")
geo_df[geo_df['Reason']=='EMS'].plot(ax = ax, markersize=20, color = "green", marker = "+",label = "Medical")
geo_df[geo_df['Reason']=='Traffic'].plot(ax = ax, markersize=20, color = "blue", marker = "o",label = "Traffic")
plt.legend(prop = {'size' : 15})
plt.title("Distribution of all 5000 distress calls")
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax,alpha = 0.4, color = "grey")
geo_df[geo_df['Reason']=='Fire'].plot(ax = ax, markersize=20, color = "orange", marker = "*",label = "Fire")
#geo_df[geo_df['Reason']=='EMS'].plot(ax = ax, markersize=20, color = "green", marker = "+",label = "Medical")
#geo_df[geo_df['Reason']=='Traffic'].plot(ax = ax, markersize=20, color = "blue", marker = "o",label = "Traffic")
plt.legend(prop = {'size' : 15})
plt.title("Distribution of Fire related distress calls")
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax,alpha = 0.4, color = "grey")
#geo_df[geo_df['Reason']=='Fire'].plot(ax = ax, markersize=20, color = "orange", marker = "*",label = "Fire")
geo_df[geo_df['Reason']=='EMS'].plot(ax = ax, markersize=20, color = "red", marker = "+",label = "Medical")
#geo_df[geo_df['Reason']=='Traffic'].plot(ax = ax, markersize=20, color = "blue", marker = "o",label = "Traffic")
plt.legend(prop = {'size' : 15})
plt.title("Distribution of EMS related distress calls")
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax,alpha = 0.4, color = "grey")
#geo_df[geo_df['Reason']=='Fire'].plot(ax = ax, markersize=20, color = "orange", marker = "*",label = "Fire")
#geo_df[geo_df['Reason']=='EMS'].plot(ax = ax, markersize=20, color = "green", marker = "+",label = "Medical")
geo_df[geo_df['Reason']=='Traffic'].plot(ax = ax, markersize=20, color = "blue", marker = "o",label = "Traffic")
plt.legend(prop = {'size' : 15})
plt.title("Distribution of Traffic related distress calls")
dayHour=df.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()
dayHour_1  = df_1.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()
dayHour_2  = df_2.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()
dayHour_3  = df_3.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis');
plt.title("Hour vs day of the week busy-ness")
#Thursday 3-5 a lot of calls are reported
plt.figure(figsize=(12,6));
sns.clustermap(dayHour,cmap='viridis');
plt.title("Cluster map distribution busy-ness per hour vs per day ")
plt.figure(figsize=(12,6));
sns.clustermap(dayHour_1,cmap='viridis');
plt.title("Cluster map distribution busy-ness per hour vs per day for EMS")
plt.figure(figsize=(12,6));
sns.clustermap(dayHour_2,cmap='viridis');
plt.title("Cluster map distribution busy-ness per hour vs per day for Fire")
plt.figure(figsize=(12,6));
sns.clustermap(dayHour_3,cmap='viridis');
plt.title("Cluster map distribution busy-ness per hour vs per day for Traffic")
dayMonth=df.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()
dayMonth_1 = df_1.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()
dayMonth_2 = df_2.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()
dayMonth_3 = df_3.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()
plt.figure(figsize=(12,6));
sns.clustermap(dayHour_1,cmap='coolwarm');
plt.title("Cluster map distribution busy-ness day of the week vs hour ")
plt.figure(figsize=(12,6));
sns.clustermap(dayMonth_1,cmap='coolwarm');
plt.title("Cluster map distribution busy-ness per month vs day of the week for EMS")
plt.figure(figsize=(12,6));
sns.clustermap(dayMonth_2,cmap='coolwarm');
plt.title("Cluster map distribution busy-ness per month vs day of the week for Fire")
plt.figure(figsize=(12,6));
sns.clustermap(dayMonth_3,cmap='coolwarm');
plt.title("Cluster map distribution busy-ness per month vs day of the week for Traffic")