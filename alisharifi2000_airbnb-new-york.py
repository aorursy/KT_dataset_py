import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import urllib

from geopy.distance import geodesic
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head(5)
len(data)
data[["price","latitude","longitude","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]].describe()
data.isnull().sum()
data.price.describe()
data = data[data['price']>0]

data.head(5)
data['price'].describe()
len(data)
data1 = data[data['price']<= data['price'].mean() + 3*data['price'].std()]

data1.head(5)
len(data1)
data1.price.describe()
data1.groupby('neighbourhood_group')['id'].agg(['count'])
data1.groupby('neighbourhood_group')['id'].agg(['count']).plot(kind="bar")
plt.figure(figsize=(15,15))

nyc_img=plt.imread(urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG'))

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

sns.scatterplot(x=data1['longitude'], y=data1['latitude'], hue='neighbourhood_group',s=20, data=data1)

plt.legend()

plt.show()
sns.set(rc={'figure.figsize':(15,10)})

sns.distplot(data1['price'],kde_kws={"label": 'price'}, bins=20)
sns.set(rc={'figure.figsize':(15,10)})

for groups in data1.neighbourhood_group.unique():

    sns.distplot(data1.price[data1['neighbourhood_group']==groups],kde_kws={"label": groups}, bins=20)
f = plt.figure(figsize=(10,30))

for i, groups in enumerate(data1.neighbourhood_group.unique()):

    f.add_subplot(5, 1, i+1)

    sns.distplot(data1.price[data1['neighbourhood_group']==groups],color="r" ,kde_kws={"label": groups}, bins=20)
data1['price_log_e'] = np.log(data1['price'])

data1.head(5)
sns.set(rc={'figure.figsize':(15,10)})

sns.distplot(data1['price_log_e'],kde_kws={"label": 'price in log e'}, bins=20)
stats.normaltest(data1["price_log_e"])
data1['price_log_10'] = np.log10(data1['price'])

data1.head(5)
sns.set(rc={'figure.figsize':(15,10)})

sns.distplot(data1['price_log_10'],kde_kws={"label": 'price in log 10'}, bins=20)
stats.normaltest(data1["price_log_10"])
sns.set(rc={'figure.figsize':(15,10)})

for groups in data1.neighbourhood_group.unique():

    sns.distplot(data1.price_log_10[data1['neighbourhood_group']==groups],kde_kws={"label": groups}, bins=20)
f = plt.figure(figsize=(10,30))

for i, groups in enumerate(data1.neighbourhood_group.unique()):

    f.add_subplot(5, 1, i+1)

    sns.distplot(data1.price_log_10[data1['neighbourhood_group']==groups],color="r" ,kde_kws={"label": groups}, bins=20)

    print('Test for Normal Distribution for ' , groups)

    print('------------------------------------------')

    print(stats.normaltest(data1["price_log_10"]))

    print('------------------------------------------')
plt.figure(figsize=(15,8))

sns.violinplot("neighbourhood_group", "price_log_10", data=data1)
plt.figure(figsize=(15,8))

sns.boxplot("neighbourhood_group", "price_log_10", data=data1)
fstat, pval = stats.f_oneway(*[data1.price_log_10[data1.neighbourhood_group == s]

for s in data1.neighbourhood_group.unique()])

print("Oneway Anova log10(price) ~ neighbourhood_group F=%.2f, p-value=%E" % (fstat, pval))
data1[["neighbourhood_group",'price']].groupby("neighbourhood_group").describe()
plt.figure(figsize=(15,8))

sns.violinplot("room_type", "price_log_10", data=data1)
plt.figure(figsize=(15,8))

sns.boxplot("room_type", "price_log_10", data=data1)
fstat, pval = stats.f_oneway(*[data1.price_log_10[data1.room_type == s]

for s in data1.room_type.unique()])

print("Oneway Anova log10(price) ~ room_type F=%.2f, p-value=%E" % (fstat, pval))
room = data1.groupby('room_type')['id'].agg(['count'])

room.head()
room.reset_index(level=0, inplace=True)

room.head()
room = room[['room_type','count']]
plt.pie(

    room['count'],

    labels=room['room_type'],

    shadow=False,

    startangle=90,

    autopct='%1.1f%%',

    )



plt.axis('equal')



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,15))

nyc_img=plt.imread(urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG'))

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

sns.scatterplot(x=data1['longitude'], y=data1['latitude'], hue='room_type',s=20, data=data1)

plt.legend()

plt.show()
data1.groupby('room_type')['id'].agg(['count']).plot(kind="bar")
data1.head(5)
data1['All_year_availability'] = 0
for i in data1.index.values:

    if data1['availability_365'][i] == 365:

        data1['All_year_availability'][i] = 1
data1.head(5)
plt.figure(figsize=(15,15))

nyc_img=plt.imread(urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG'))

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

sns.scatterplot(x=data1['longitude'], y=data1['latitude'], hue='All_year_availability',s=20, data=data1)

plt.legend()

plt.show()
sns.violinplot("neighbourhood_group", "price_log_10", hue="All_year_availability",data=data1)
sns.boxplot("neighbourhood_group", "price_log_10", hue="All_year_availability",data=data1)
sns.violinplot("neighbourhood_group", "price_log_10", hue="All_year_availability",data=data1 , split=True)
sns.violinplot("All_year_availability", "price_log_10",data=data1)
sns.violinplot("All_year_availability", "price_log_10", hue="neighbourhood_group",data=data1)
sns.boxplot("All_year_availability", "price_log_10", hue="neighbourhood_group",data=data1)
sns.boxplot("All_year_availability", "price_log_10",data=data1)
stats.ttest_rel(data1['All_year_availability'],data1["price_log_10"])
data1['minimum_nights'].describe()
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","price_log_10", hue="neighbourhood_group", data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","price_log_10", hue="room_type", data=data1)
sns.lmplot("minimum_nights", "price_log_10", hue="All_year_availability",data=data1)
data1.isnull().sum()
data1[data1['reviews_per_month'].isnull()]
data1[['reviews_per_month','neighbourhood_group']] = data1[['reviews_per_month','neighbourhood_group']].fillna(value=0)
data1.reviews_per_month.describe()
data1.isnull().sum()
plt.figure(i,figsize=(20,15))

correlation_matrix = data1[["price",'price_log_10',"minimum_nights","All_year_availability","number_of_reviews","reviews_per_month"]].corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
temp = data1[data1['reviews_per_month']>0]

temp.head(5)
temp.describe()
plt.figure(figsize=(15,8))

sns.violinplot("neighbourhood_group", "reviews_per_month", data=temp)
plt.figure(figsize=(15,10))

sns.boxplot("neighbourhood_group", "reviews_per_month", data=temp)
fstat, pval = stats.f_oneway(*[temp.reviews_per_month[temp.neighbourhood_group == s]

for s in temp.neighbourhood_group.unique()])

print("Oneway Anova reviews_per_month ~ neighbourhood_group F=%.2f, p-value=%E" % (fstat, pval))
plt.figure(i,figsize=(20,15))

correlation_matrix = temp[["price",'price_log_10',"minimum_nights","All_year_availability","number_of_reviews","reviews_per_month"]].corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
stats.stats.spearmanr(data1['reviews_per_month'],data1['price_log_10'])
data1.head(5)
data1.reset_index()
duplicate_host = data1[data1[['host_id']].duplicated()]

duplicate_host.head(5)
duplicate = duplicate_host.groupby('host_id')['id'].agg(['count']).sort_values(by=['count'],ascending=False)

duplicate.head(25)
duplicate.reset_index(level=0, inplace=True)
frames = []

counter = 0

for ids in duplicate["host_id"]:

    

    

    group_list = []

    num_list = []

    

    temp = data1[data1['host_id'] == ids]

    price = temp.price.sum()

    avarage = price/len(temp)

    t = temp.groupby('neighbourhood_group')['id'].agg(['count'])

    t.reset_index(level=0, inplace=True)

    for i in range(len(t)):

        group_name = t["neighbourhood_group"][i]

        if group_name not in group_list:

            group_list.append(group_name)

            num_group = t['count'][i]

            num_list.append(num_group)

        

    if 'Brooklyn' not in group_list:

        group_list.append("Brooklyn")

        num_list.append(0)

    if 'Manhattan' not in group_list:

        group_list.append("Manhattan")

        num_list.append(0)

    if 'Queens' not in group_list:

        group_list.append("Queens")

        num_list.append(0)

    if 'Staten Island' not in group_list:

        group_list.append("Staten Island")

        num_list.append(0)

    if 'Bronx' not in group_list:

        group_list.append("Bronx")

        num_list.append(0)

        

    dict = {'host_id':[ids],'total_price':[price],'average_price':[avarage],'num':[len(temp)]}

    for i in range(5):

        d = {group_list[i]:[num_list[i]]}

        dict.update(d)

    df_temp = pd.DataFrame(dict,index=[counter])   

    frames.append(df_temp)

    counter = counter + 1

duplicate_price = pd.concat(frames)

duplicate_price.head(15)
nyc_metro = pd.read_csv('../input/newyorkcityairbnblocations/NYC_Transit_Subway_Entrance_And_Exit_Data.csv')

nyc_metro.head(5)
nyc_metro_location = nyc_metro[['Entrance Latitude','Entrance Longitude']]

nyc_metro_location.head(5)
nyc_metro_location.describe()
data1 = data1.reset_index(drop=True)
data1.head(5)
# data1['distance_metro_entrance'] = 0.0
# for i in range(len(data1)):

#     Latitude = data1['latitude'][i]

#     Longitude = data1['longitude'][i]

#     min_distance = 9999999

#     for j in range(len(nyc_metro_location)):

#         metro_Latitude = nyc_metro_location['Entrance Latitude'][j]

#         metro_Longitude = nyc_metro_location['Entrance Longitude'][j]

#         origin = (Latitude, Longitude)

#         dist = (metro_Latitude, metro_Longitude)

#         distance = geodesic(origin, dist).meters

#         if distance <= min_distance:

#             min_distance = distance

#     data1['distance_metro_entrance'][i] = round(min_distance, 2)

#     if i%100 == 0:

#         print(i,round(min_distance, 2))
#data1.head(5)
# data1.to_csv('distance_metro.csv')
metro_distance = pd.read_csv('../input/newyorkcityairbnblocations/distance_metro.csv')
metro_distance.head(5)
metro_distance['distance_metro_entrance'].describe()
plt.figure(figsize=(15,10))

sns.boxplot("neighbourhood_group", "distance_metro_entrance", data=metro_distance)
plt.figure(figsize=(15,8))

sns.violinplot("neighbourhood_group", "distance_metro_entrance", data=metro_distance)
plt.figure(i,figsize=(20,15))

correlation_matrix = metro_distance[["price",'price_log_10',"minimum_nights","All_year_availability","number_of_reviews","reviews_per_month","distance_metro_entrance"]].corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
stats.stats.spearmanr(metro_distance['distance_metro_entrance'],metro_distance['price_log_10'])
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_metro_entrance", hue="neighbourhood_group", data=metro_distance)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_metro_entrance", hue="room_type", data=metro_distance)
stats.stats.spearmanr(metro_distance['distance_metro_entrance'],metro_distance['minimum_nights'])
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","distance_metro_entrance", hue="neighbourhood_group", data=metro_distance)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","distance_metro_entrance", hue="room_type", data=metro_distance)
f = plt.figure(figsize=(10,30))

for i, groups in enumerate(metro_distance.neighbourhood_group.unique()):

    f.add_subplot(5, 1, i+1)

    sns.distplot(metro_distance.distance_metro_entrance[metro_distance['neighbourhood_group']==groups],color="r" ,kde_kws={"label": groups}, bins=20)

    print('Test for Normal Distribution for ' , groups)

    print('------------------------------------------')

    print(stats.normaltest(metro_distance["distance_metro_entrance"]))

    print('------------------------------------------')
#data1['distance_from_central_station'] = 0.0
#data1.head(5)
# central_station_latitude = 40.7527

# central_station_longitude = -73.9772

# origin = (central_station_latitude,central_station_longitude)

# for i in range(len(data1)):

#     Latitude = data1['latitude'][i]

#     Longitude = data1['longitude'][i]

#     dist = (Latitude,Longitude)

#     min_distance = geodesic(origin, dist).meters

#     data1['distance_from_central_station'][i] = round(min_distance, 2)

#     if i%100 == 0:

#         print(i,round(min_distance, 2))
#data1.to_csv('distance_from_central_station.csv')
data1 = pd.read_csv('../input/newyorkcityairbnbdistance/distance_from_central_station.csv')
plt.figure(figsize=(15,10))

sns.boxplot("neighbourhood_group", "distance_from_central_station", data=data1)
plt.figure(figsize=(15,8))

sns.violinplot("neighbourhood_group", "distance_from_central_station", data=data1)
plt.figure(figsize=(15,8))

sns.violinplot("room_type", "distance_from_central_station", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("room_type", "distance_from_central_station", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("All_year_availability", "distance_from_central_station", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("All_year_availability", "distance_from_central_station",hue = 'room_type', data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("All_year_availability", "distance_from_central_station",hue = 'neighbourhood_group', data=data1)
stats.stats.spearmanr(data1['distance_from_central_station'],data1['price_log_10'])
plt.figure(i,figsize=(20,15))

correlation_matrix = data1[["price",'price_log_10',"minimum_nights","All_year_availability","number_of_reviews","reviews_per_month","distance_from_central_station"]].corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_from_central_station", hue="neighbourhood_group", data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_from_central_station",line_kws={'color': 'red'}, data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_from_central_station", hue="room_type", data=data1)
f = plt.figure(figsize=(10,30))

for i, groups in enumerate(data1.neighbourhood_group.unique()):

    f.add_subplot(5, 1, i+1)

    sns.distplot(data1.distance_from_central_station[data1['neighbourhood_group']==groups],color="r" ,kde_kws={"label": groups}, bins=20)

    print('Test for Normal Distribution for ' , groups)

    print('------------------------------------------')

    print(stats.normaltest(data1["distance_from_central_station"]))

    print('------------------------------------------')
data1['distance_from_JFK_airport'] = 0.0
data1.head(5)
JFK_airport_latitude = 40.6413

JFK_airport_longitude = -73.7781

origin = (JFK_airport_latitude,JFK_airport_longitude)

for i in range(len(data1)):

    Latitude = data1['latitude'][i]

    Longitude = data1['longitude'][i]

    dist = (Latitude,Longitude)

    min_distance = geodesic(origin, dist).meters

    data1['distance_from_JFK_airport'][i] = round(min_distance, 2)

    if i%100 == 0:

        print(i,round(min_distance, 2))
plt.figure(figsize=(15,10))

sns.boxplot("neighbourhood_group", "distance_from_JFK_airport", data=data1)
plt.figure(figsize=(15,8))

sns.violinplot("neighbourhood_group", "distance_from_JFK_airport", data=data1)
plt.figure(figsize=(15,8))

sns.violinplot("room_type", "distance_from_JFK_airport", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("room_type", "distance_from_JFK_airport", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("All_year_availability", "distance_from_JFK_airport", data=data1)
plt.figure(figsize=(15,10))

sns.boxplot("All_year_availability", "distance_from_JFK_airport",hue = 'room_type', data=data1)
f = plt.figure(figsize=(10,30))

for i, groups in enumerate(data1.neighbourhood_group.unique()):

    f.add_subplot(5, 1, i+1)

    sns.distplot(data1.distance_from_JFK_airport[data1['neighbourhood_group']==groups],color="r" ,kde_kws={"label": groups}, bins=20)

    print('Test for Normal Distribution for ' , groups)

    print('------------------------------------------')

    print(stats.normaltest(data1["distance_from_JFK_airport"]))

    print('------------------------------------------')
stats.stats.spearmanr(data1['distance_from_JFK_airport'],data1['minimum_nights'])
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","distance_from_JFK_airport", hue="neighbourhood_group", data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","distance_from_JFK_airport", hue="room_type", data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("minimum_nights","distance_from_JFK_airport", hue="All_year_availability", data=data1)
stats.stats.spearmanr(data1['distance_from_JFK_airport'],data1['price_log_10'])
plt.figure(i,figsize=(20,15))

correlation_matrix = data1[["price",'price_log_10',"minimum_nights","All_year_availability","number_of_reviews","reviews_per_month","distance_from_JFK_airport"]].corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_from_JFK_airport", hue="neighbourhood_group", data=data1)
sns.set(rc={'figure.figsize':(30,20)})

sns.lmplot("price_log_10","distance_from_JFK_airport", hue="room_type", data=data1)