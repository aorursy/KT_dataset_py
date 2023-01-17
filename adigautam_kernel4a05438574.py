# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/australian-bush-fire-satellite-data-nasa/fire_nrt_V1_101674.csv')
df1=pd.read_csv('/kaggle/input/australian-bush-fire-satellite-data-nasa/fire_archive_V1_101674.csv')
df2=pd.read_csv('/kaggle/input/australian-bush-fire-satellite-data-nasa/fire_archive_M6_101673.csv')
df3=pd.read_csv('/kaggle/input/australian-bush-fire-satellite-data-nasa/fire_nrt_M6_101673.csv')
df_merged = pd.concat([df1,df],sort=True)
data = df_merged
data.head()
data.info()
df_new = data[["latitude","longitude","acq_date","frp"]]
df_new.head()
df=df_new[df_new['acq_date']>='2019-11-01']
df.head()
df_topaffected=df.sort_values(by='frp',ascending=False)
df_topaffected.head(10)
df_dates=df_topaffected[['acq_date','frp']].set_index('acq_date')
df_dates=df_dates.groupby('acq_date').sum().sort_values('frp',ascending=False)
df_dates.head()
sns.distplot(dfx['frp'])
import seaborn as sns
df_merged['frp'].max()
dfd=df_new.sort_values('acq_date')
x=dfd['frp'].max()
dfd[dfd['frp']==x]


sns.barplot(x='acq_date',y='frp',data=dfd)
#Create a map
import folium
m = folium.Map(location=[-35.0,144], control_scale=True, zoom_start=4,attr = "text some",)
df_copy = df_topaffected.copy()

# loop through data to create Marker for each hospital
for i in range(0,200):
     
    folium.Marker(location=[df_copy.iloc[i]['latitude'], df_copy.iloc[i]['longitude']],tooltip="frp: " + str(df_copy.iloc[i]['frp']) + "<br/> date: "+ str(df_copy.iloc[i]['acq_date']),icon=folium.Icon(color='red',icon='fire',prefix="fa"),
    ).add_to(m)
        
m
from folium.plugins import HeatMapWithTime
# A small function to get heat map with time given the data

def getmap(ip_data,location,zoom,radius):
    
    #get day list
    dfmap = ip_data[['acq_date','latitude','longitude','frp']]
    df_day_list = []
    for day in dfmap.acq_date.sort_values().unique():
        df_day_list.append(dfmap.loc[dfmap.acq_date == day, ['acq_date','latitude', 'longitude', 'frp']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist())
    
    # Create a map using folium
    m = folium.Map(location, zoom_start=zoom,tiles='Stamen Terrain')
    #creating heatmap with time
    HeatMapWithTime(df_day_list,index =list(dfmap.acq_date.sort_values().unique()), auto_play=False,radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(m)

    return m
getmap(df,[-27,132],3.5,3)

