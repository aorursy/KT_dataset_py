# 22p22c0589_Naratip_W2H3_27092020
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.info()
# เนื่องจากข้อมูลมีปริมาณเยอะและันโมเดลไม่ผ่านจึง ทำการ scope data ลงมา
# ใช้ data ของคืนขั้นต่ำ < 7วัน(คิดว่าเป็นระยะเวลาคืนขั้นต่ำที่พอรับได้มากที่สุด ในการเข้าพัก)
# และมีห้องว่าง มากกว่า 2 วันในหนึ่งปี อย่างน้อย(คิดว่า ถ้าต่ำกว่านั้นห้องพักที่น่าจะไม่รับนักท่องเที่ยวแล้วแต่ข้อมูลยังค้างในระบบ)
df = df[df['minimum_nights'] <= 7]
df = df[df['availability_365'] >= 2]
# drop แถวของ data ที่ไม่มีค่า ในคอลั้มที่ fill ค่าเข้าไปยาก
df = df.dropna(subset=['name'])
df = df.dropna(subset=['neighbourhood_group'])
# dataframe2 = dataframe.dropna(axis=0)
#check unique of neighbourhood_group
df['neighbourhood_group'].unique()

df.drop(['id', 'host_id', 'host_name','last_review','reviews_per_month','calculated_host_listings_count','availability_365','neighbourhood'], 1, inplace=True)
df = df.reset_index(drop=True)
df
#One hot encoding
df_cal = pd.get_dummies(df, columns=['room_type','neighbourhood_group'],drop_first = False)
df_cal.columns
# splite binary data to other dataframe
df_bi = df_cal[['room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room',
       'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island']]
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch # draw dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
# Standardization in some coulumn
cols = ['price','minimum_nights','number_of_reviews']
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True) # support only positive value
mat = pt.fit_transform(df_cal[cols])

# comnine data that do Standardization with data that do one hot encode
X = pd.DataFrame(mat, columns=cols)
X = pd.concat([X, df_bi], axis=1)
X
# hierarchical clustering
hc = AgglomerativeClustering(n_clusters=6, linkage='ward')
hc.fit(X)
hc.labels_
df['cluster'] = hc.labels_
df.groupby('cluster').head(3).sort_values('cluster')
# heatmap by cluster
X['cluster']=hc.labels_
fig, ax = plt.subplots(figsize=(20, 4))

sns.heatmap(X.groupby('cluster').median(), cmap="Blues", linewidths=1, 
            square=True, annot=True, fmt='.2f', ax=ax ,vmin=0, vmax=1);
# Dendrogram
fig, ax=plt.subplots(figsize=(60, 40))
dg=sch.dendrogram(sch.linkage(X[:3000], method='ward'), ax=ax, labels=df['name'].values)
# Earth map
import folium
from folium.plugins import MarkerCluster

data = df[:1000]
details_col = 'name'.split()
x = 'latitude'
y = 'longitude'

world_map_final = folium.Map(location=[40.7128,-74.0060 ],tiles='cartodbpositron',zoom_start=11) 
# world_map= folium.Map(tiles="OpenStreetMap")

for i in range(len(data)):
    lat = data.iloc[i][x]
    long = data.iloc[i][y]
    radius = 4
    popup_text = """{}<br>"""
    popup_text = popup_text.format(df[details_col].iloc[i])

    if df['cluster'][i] == 0:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='blue').add_to(world_map_final)
    elif df['cluster'][i] == 1:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='lightblue').add_to(world_map_final)
    elif df['cluster'][i] == 5:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='darkblue').add_to(world_map_final)
    elif df['cluster'][i] == 2:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='green').add_to(world_map_final)
    elif df['cluster'][i] == 3:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='lightgreen').add_to(world_map_final)
    elif df['cluster'][i] == 4:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='yellow').add_to(world_map_final)

world_map_final
