# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime 

import seaborn as sb

import matplotlib.pyplot as plt

import folium

import scipy as scp



from sklearn import feature_selection as fs

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from folium.plugins import MarkerCluster

import category_encoders as ce



import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ny_data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',index_col = 'host_id')

ny_data.info()


print(ny_data[ny_data['number_of_reviews']==0][['number_of_reviews','last_review','reviews_per_month']]);
ny_data['reviews_per_month'].fillna(0,inplace = True)

today = datetime.datetime(2019,7,8)

ny_data['last_review'].fillna(today,inplace = True)



ny_data['days_since_last_review'] = (pd.to_datetime(today) - pd.to_datetime(ny_data['last_review']))/np.timedelta64(1,'D')
print(ny_data.info())
ny_data = ny_data.drop(['name','id','host_name','last_review'],axis = 1)

ny_data.head()
plt.figure(figsize = (18,18));

plt.subplot(221);

counts = ny_data['room_type'].value_counts();

plt.pie(counts,labels = counts.index,colors=['darkcyan','skyblue','powderblue'],autopct='%1.2f%%');

plt.title('Distribution of data with respect to room type');



plt.subplot(222);

sb.countplot(data = ny_data ,x= 'neighbourhood_group',color = 'cadetblue');

plt.title('Distribution of data with respect to neighbourhood group');

sb.despine(offset = 10,left = True,bottom = True);



plt.subplot(223);

plt.hist(ny_data['price'],bins =60,color = 'cadetblue');

plt.title('Distribution of data with respect to price');

plt.xlim(0,2500);

plt.xlabel('price $');

plt.ylabel('count');



plt.subplot(224);

plt.hist(ny_data['availability_365'],bins =20,color='cadetblue');

plt.title('Distribution of data with respect to availabilty');

plt.xlabel('avaliability');

plt.ylabel('count');



sb.despine(left=True,bottom=True);
brooklyn_data = ny_data.loc[ny_data['neighbourhood_group']=='Brooklyn']



map_brooklyn=folium.Map(location=[40.638177,-73.964160],tiles="stamenterrain",zoom_start=11)

marker_cluster = MarkerCluster().add_to(map_brooklyn)



locations = brooklyn_data[['latitude', 'longitude']]

locationlist = locations.values.tolist()



for point in range(0, len(locationlist)):

    folium.Marker(locationlist[point]).add_to(marker_cluster)

    

map_brooklyn
fig = plt.figure(figsize=(18,12))

fig.add_subplot(1,2,1)

g=sb.scatterplot(brooklyn_data.longitude,brooklyn_data.latitude,hue=brooklyn_data.room_type,palette='deep',alpha =0.6)

g.set_title('Distribution of listings based on room type according to lat-long location points');

sb.despine(offset = 10,trim=True,ax=g);

fig.add_subplot(2,2,2)

x=sb.violinplot(x=brooklyn_data.room_type,y=brooklyn_data.price,palette='deep',inner="quartile");

plt.ylim(-100,500);

plt.title('Price distribution according to room type');

sb.despine(offset = 10,bottom = True,left=True,ax=x);

fig.add_subplot(2,2,4);

y=sb.violinplot(x=brooklyn_data.room_type,y=brooklyn_data.availability_365,palette='deep',inner="quartile");

sb.despine(offset = 10,bottom = True,left=True,ax=y);

plt.title('avaliability distribution according to room type');
new_data=brooklyn_data[brooklyn_data.columns[4:]] 

new_data=new_data[new_data['price']<2100]

y=new_data.columns

count=0

plt.figure(figsize=(18,18));

for i in range(2,8):

    plt.subplot(3,3,count+1);

    sb.scatterplot(data=new_data,x='price',y=y[i],hue='room_type',palette='deep');

    count=count+1

sb.despine(offset = 10,bottom = True,left=True);
brooklyn_data = ny_data.loc[ny_data['neighbourhood_group']=='Brooklyn']

plt.figure(figsize = (12,14));



plt.subplot(221);

new_data = brooklyn_data.groupby('neighbourhood')['price'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'price':new_data.values}).sort_values(by=['price']).head(10);

barlist=plt.barh(new_data['neighbourhood'],new_data['price'],color = 'cadetblue');

for i, v in enumerate(new_data['price']):

    plt.text(v + 1,i-0.3, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.xticks([]);

barlist[7].set_color('lightcoral');

plt.title('Best Places in Brooklyn by average price $');





plt.subplot(222);

new_data = brooklyn_data.groupby('neighbourhood')['availability_365'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'availability_365':new_data.values}).sort_values(by=['availability_365']).tail(10);

barlist=plt.barh(new_data['neighbourhood'],new_data['availability_365'],color = 'cadetblue');

for i, v in enumerate(new_data['availability_365']):

    plt.text(v + 1,i-0.3, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.xticks([]);

barlist[4].set_color('lightcoral');

plt.title('Best Places in Brooklyn by average availability in days');



plt.subplot(212);

new_data = brooklyn_data.groupby('neighbourhood')['number_of_reviews'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'number_of_reviews':new_data.values}).sort_values(by=['number_of_reviews']).tail(10);

barlist = plt.bar(new_data['neighbourhood'],new_data['number_of_reviews'],color = 'cadetblue');

for i, v in enumerate(new_data['number_of_reviews']):

    plt.text(i-0.3,v + 1, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.yticks([]);

barlist[1].set_color('lightcoral');

plt.title('Best Places in Brooklyn by average no of reviews');

plt.tight_layout()
queens_data = ny_data.loc[ny_data['neighbourhood_group']=='Queens']



map_queens=folium.Map(location=[40.6582,-73.7949],tiles="stamenterrain",zoom_start=11)

marker_cluster = MarkerCluster().add_to(map_queens)



locations = queens_data[['latitude', 'longitude']]

locationlist = locations.values.tolist()



for point in range(0, len(locationlist)):

    folium.Marker(locationlist[point]).add_to(marker_cluster)

    

map_queens
fig = plt.figure(figsize=(18,12))

fig.add_subplot(1,2,1)

g=sb.scatterplot(queens_data.longitude,queens_data.latitude,hue=queens_data.room_type,palette='deep',alpha =0.6)

g.set_title('Distribution of listings based on room type according to lat-long location points');

sb.despine(offset = 10,trim=True,ax=g);

fig.add_subplot(2,2,2)

x=sb.violinplot(x=queens_data.room_type,y=queens_data.price,palette='deep',inner="quartile");

plt.ylim(-100,500);

plt.title('Price distribution according to room type');

sb.despine(offset = 10,bottom = True,left=True,ax=x);

fig.add_subplot(2,2,4);

y=sb.violinplot(x=queens_data.room_type,y=queens_data.availability_365,palette='deep',inner="quartile");

sb.despine(offset = 10,bottom = True,left=True,ax=y);

plt.title('avaliability distribution according to room type');
new_data=queens_data[queens_data.columns[4:]] 

new_data=new_data[new_data['price']<2100]

y=new_data.columns

count=0

plt.figure(figsize=(18,18));

for i in range(2,8):

    plt.subplot(3,3,count+1);

    sb.scatterplot(data=new_data,x='price',y=y[i],hue='room_type',palette='deep');

    count=count+1

sb.despine(offset = 10,bottom = True,left=True);
brooklyn_data = ny_data.loc[ny_data['neighbourhood_group']=='Queens']

plt.figure(figsize = (12,18));



plt.subplot(221);

new_data = brooklyn_data.groupby('neighbourhood')['price'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'price':new_data.values}).sort_values(by=['price']).head(10);

barlist=plt.barh(new_data['neighbourhood'],new_data['price'],color = 'cadetblue');

for i, v in enumerate(new_data['price']):

    plt.text(v + 1,i-0.3, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.xticks([]);

barlist[9].set_color('lightcoral');

plt.title('Best Places in Queens by average price $');





plt.subplot(222);

new_data = brooklyn_data.groupby('neighbourhood')['availability_365'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'availability_365':new_data.values}).sort_values(by=['availability_365']).tail(10);

barlist=plt.barh(new_data['neighbourhood'],new_data['availability_365'],color = 'cadetblue');

for i, v in enumerate(new_data['availability_365']):

    plt.text(v + 1,i-0.3, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.xticks([]);

barlist[5].set_color('lightcoral');

plt.title('Best Places in Queens by average availability in days');



plt.subplot(212);

new_data = brooklyn_data.groupby('neighbourhood')['number_of_reviews'].mean();

new_data = pd.DataFrame({'neighbourhood':new_data.index,'number_of_reviews':new_data.values}).sort_values(by=['number_of_reviews']).tail(10);

barlist = plt.bar(new_data['neighbourhood'],new_data['number_of_reviews'],color = 'cadetblue');

for i, v in enumerate(new_data['number_of_reviews']):

    plt.text(i-0.3,v + 1, str(round(v,2)), color='grey', fontweight='bold');

sb.despine(left=True,bottom=True);

plt.yticks([]);

barlist[7].set_color('lightcoral');

plt.xticks(rotation=15);

plt.title('Best Places in Queens by average no of reviews');

plt.tight_layout()