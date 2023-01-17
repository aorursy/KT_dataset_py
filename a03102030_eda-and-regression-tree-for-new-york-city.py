import numpy as np

import pandas as pd 

import geopandas as gpd

import matplotlib.pyplot as plt

import os

import seaborn as sns

import folium

from folium import plugins
os.listdir('../input/new-york-city-airbnb-open-data/')
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
data.isnull().sum()
data=data.dropna(subset=['name'])
data.neighbourhood_group.unique()
fig,ax=plt.subplots(1,2,figsize=(20,6))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data.neighbourhood_group.value_counts().plot.bar(color=clr,ax=ax[0])

ax[0].set_title('The number of rooms in each neighbourhood_group',size=20)

ax[0].set_ylabel('rooms',size=18)

ax[0].tick_params(axis='x', rotation=360)

ax[0].tick_params(labelsize=18)



data.groupby(['neighbourhood_group','room_type'])['id'].agg('count').unstack('room_type').plot.bar(ax=ax[1])

ax[1].tick_params(axis='x', rotation=360)

ax[1].set_title('The number of rooms in each room_type',size=20)

ax[1].set_ylabel('rooms',size=18)

ax[1].set_xlabel('')

ax[1].tick_params(labelsize=18)
data.room_type.unique()
fig,ax=plt.subplots(3,1,figsize=(15,36))

sns.boxplot(x="neighbourhood_group", y="price", data=data[data.room_type=='Shared room'],ax=ax[0])

ax[0].set_title("Boxplot of Price for 'Shared room' in each neighbourhood_group",size=20)



sns.boxplot(x="neighbourhood_group", y="price", data=data[data.room_type=='Private room'],ax=ax[1])

ax[1].set_title("Boxplot of Price for 'Private room' in each neighbourhood_group",size=20)



sns.boxplot(x="neighbourhood_group", y="price", data=data[data.room_type=='Entire home/apt'],ax=ax[2])

ax[2].set_title("Boxplot of Price for 'Entire home/apt' in each neighbourhood_group",size=20)
data_manha=data[data.neighbourhood_group=='Manhattan']

data_manha.head()
data_manha.neighbourhood.unique()
fig,ax=plt.subplots(1,2,figsize=(15,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data_manha.neighbourhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 neighbourhood by the number of rooms",size=20)

ax[0].set_xlabel('rooms',size=18)





count=data_manha['neighbourhood'].value_counts()

groups=list(data_manha['neighbourhood'].value_counts().index)[:10]

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 

plt.subplots_adjust(wspace =0.5, hspace =0)

plt.ylabel('')
data_manha_65=data_manha[data_manha.price<65]

data_manha_65['label']=data_manha_65.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

data_manha_65.head()
Long=-73.92

Lat=40.86

manha_map=folium.Map([Lat,Long],zoom_start=12)



manha_rooms_map=plugins.MarkerCluster().add_to(manha_map)

for lat,lon,label in zip(data_manha_65.latitude,data_manha_65.longitude,data_manha_65.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(manha_rooms_map)

manha_map.add_child(manha_rooms_map)



manha_map
data_manha_65_80=data_manha.loc[(data_manha['price'] >=65) & (data_manha['price'] <80)]

data_manha_65_80['label']=data_manha_65_80.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

Long=-73.92

Lat=40.86

manha_65_80_map=folium.Map([Lat,Long],zoom_start=12)



manha_65_80_rooms_map=plugins.MarkerCluster().add_to(manha_65_80_map)

for lat,lon,label in zip(data_manha_65_80.latitude,data_manha_65_80.longitude,data_manha_65_80.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(manha_65_80_rooms_map)

manha_65_80_map.add_child(manha_65_80_rooms_map)



manha_65_80_map
data_Brooklyn=data[data.neighbourhood_group=='Brooklyn']

data_Brooklyn.head()
data_Brooklyn.neighbourhood.unique()
fig,ax=plt.subplots(1,2,figsize=(15,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data_Brooklyn.neighbourhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 neighbourhood by the number of rooms",size=20)

ax[0].set_xlabel('rooms',size=18)





count=data_Brooklyn['neighbourhood'].value_counts()

groups=list(data_Brooklyn['neighbourhood'].value_counts().index)[:10]

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 

plt.subplots_adjust(wspace =0.5, hspace =0)

plt.ylabel('')
data_Brooklyn_10_65=data_Brooklyn.loc[(data_Brooklyn['price'] >=10) & (data_Brooklyn['price'] <65)][:2000]

data_Brooklyn_10_65['label']=data_Brooklyn_10_65.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

data_Brooklyn_10_65.head()
Long=-73.94

Lat=40.72

Brooklyn_10_65_map=folium.Map([Lat,Long],zoom_start=12)



Brooklyn_10_65_rooms_map=plugins.MarkerCluster().add_to(Brooklyn_10_65_map)

for lat,lon,label in zip(data_Brooklyn_10_65.latitude,data_Brooklyn_10_65.longitude,data_Brooklyn_10_65.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(Brooklyn_10_65_rooms_map)

Brooklyn_10_65_map.add_child(Brooklyn_10_65_rooms_map)



Brooklyn_10_65_map
data_Queens=data[data.neighbourhood_group=='Queens']

data_Queens.head()
data_Queens.neighbourhood.unique()
fig,ax=plt.subplots(1,2,figsize=(15,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data_Queens.neighbourhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 neighbourhood by the number of rooms",size=20)

ax[0].set_xlabel('rooms',size=18)





count=data_Queens['neighbourhood'].value_counts()

groups=list(data_Queens['neighbourhood'].value_counts().index)[:10]

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 

plt.subplots_adjust(wspace =0.5, hspace =0)

plt.ylabel('')
data_Queens_100_1=data_Queens.loc[(data_Queens['price'] <100)][:2000]

data_Queens_100_1['label']=data_Queens_100_1.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

data_Queens_100_1.head()
Long=-73.80

Lat=40.70

data_Queens_100_1_map=folium.Map([Lat,Long],zoom_start=12)



data_Queens_100_1_rooms_map=plugins.MarkerCluster().add_to(data_Queens_100_1_map)

for lat,lon,label in zip(data_Queens_100_1.latitude,data_Queens_100_1.longitude,data_Queens_100_1.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_Queens_100_1_rooms_map)

data_Queens_100_1_map.add_child(data_Queens_100_1_rooms_map)



data_Queens_100_1_map
data_Queens_100_2=data_Queens.loc[(data_Queens['price'] <100)][2000:2800]

data_Queens_100_2['label']=data_Queens_100_2.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

Long=-73.80

Lat=40.70

data_Queens_100_2_map=folium.Map([Lat,Long],zoom_start=12)



data_Queens_100_2_rooms_map=plugins.MarkerCluster().add_to(data_Queens_100_2_map)

for lat,lon,label in zip(data_Queens_100_2.latitude,data_Queens_100_2.longitude,data_Queens_100_2.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_Queens_100_2_rooms_map)

data_Queens_100_2_map.add_child(data_Queens_100_2_rooms_map)



data_Queens_100_2_map
data_Queens_100_3=data_Queens.loc[(data_Queens['price'] <100)][2801:-1]

data_Queens_100_3['label']=data_Queens_100_3.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

Long=-73.80

Lat=40.70

data_Queens_100_3_map=folium.Map([Lat,Long],zoom_start=12)



data_Queens_100_3_rooms_map=plugins.MarkerCluster().add_to(data_Queens_100_3_map)

for lat,lon,label in zip(data_Queens_100_3.latitude,data_Queens_100_3.longitude,data_Queens_100_3.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_Queens_100_3_rooms_map)

data_Queens_100_3_map.add_child(data_Queens_100_3_rooms_map)



data_Queens_100_3_map
fig,ax=plt.subplots(1,2,figsize=(18,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data.groupby(['host_name'])['number_of_reviews'].agg('sum').sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 host by the number of reviews",size=20)

ax[0].set_xlabel('reviews',size=18)

ax[0].set_ylabel('')



data.groupby(['host_name'])['price'].agg('mean').sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[1])

ax[1].set_title("Top 10 host by the average of price for rooms",size=20)

ax[1].set_xlabel('average of price',size=18)

ax[1].set_ylabel('')
plt.figure(figsize=(15,15))

corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)

plt.title("correlation plot",size=28)
data_tree=data[['neighbourhood_group','neighbourhood','room_type','minimum_nights','number_of_reviews','price']]

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_tree['neighbourhood_group_new'] = labelencoder.fit_transform(data_tree['neighbourhood_group'])

data_tree['neighbourhood_new'] = labelencoder.fit_transform(data_tree['neighbourhood'])

data_tree['room_type_new'] = labelencoder.fit_transform(data_tree['room_type'])

data_tree.head()
data_tree=data_tree[data_tree.price<=180]

data_tree=data_tree[data_tree.price>=90]

len(data_tree)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

x_train,x_test,y_train,y_test=train_test_split(data_tree[['neighbourhood_group_new','neighbourhood_new','room_type_new','minimum_nights','number_of_reviews']],data_tree[['price']],test_size=0.1,random_state=0)

Reg_tree=DecisionTreeRegressor(criterion='mse',max_depth=3,random_state=0)

Reg_tree=Reg_tree.fit(x_train,y_train)

y=y_test['price']

predict=Reg_tree.predict(x_test)

print("median absolute deviation (MAD): ",np.mean(abs(np.multiply(np.array(y_test.T-predict),np.array(1/y_test)))))
from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

from IPython.display import Image as PImage

from sklearn.tree import export_graphviz

with open("tree1.dot", 'w') as f:

     f = export_graphviz(Reg_tree,

                              out_file=f,

                              max_depth = 3,

                              impurity = True,

                              feature_names = ['neighbourhood_group_new','neighbourhood_new','room_type_new','minimum_nights','number_of_reviews'],

                              rounded = True,

                              filled= True )

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")