import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns
data=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_org.csv')

data.head(8)
data.isnull().sum()
data=pd.read_csv('../input/ny_airbnb/AB_NYC_2019.csv')

data.isnull().sum()
# SELECT COUNT(*) FROM NY1 WHERE Price <= 0

# Result: 11 --- will remove those 11 rows



# SELECT COUNT(*) FROM NY1 WHERE Price <= 0

# Result: 0 --- good to go



# SELECT COUNT(*) FROM NY1 WHERE availability_365 <= 0

# Result: 17533 --- thats is shocking, more than one third of property listed has never been available
#Courtesy Dgmonov, I didn't know earler how to use image while plotting



plt.figure(figsize=(20,16))

nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group, ax=ax)
# CREATE TABLE [dbo].[NY_2](



# 	[id] [int] NULL,

#   [name] [varchar](max) NULL,

#   [host_id] [int] NULL,

# 	[host_name] [varchar](max) NULL,

# 	[neighbourhood_group] [varchar](max) NULL,

# 	[neighbourhood] [varchar](max) NULL,

# 	[latitude] [decimal](18, 9) NULL,

# 	[longitude] [decimal](18, 9) NULL,

# 	[room_type] [varchar](max) NULL,

# 	[price] [int] NULL,

# 	[minimum_nights] [int] NULL,

# 	[number_of_reviews] [int] NULL,

# 	[last_review] [datetime] NULL,

# 	[reviews_per_month] [decimal](18, 9) NULL,

# 	[calculated_host_listings_count] [int] NULL,

# 	[availability_365] [int] NULL,

# 	[Price_Range_100] [varchar](max) NULL,

# 	[Price_Range_Sequence] INT,

# 	[Availability_Range] [varchar](max) NULL,

# 	[Availibility_Range_Sequence] INT

# ) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
# INSERT INTO NY_2(id,name,host_id,host_name,neighbourhood_group,neighbourhood,latitude,longitude,room_type,price,minimum_nights,number_of_reviews,last_review,reviews_per_month,calculated_host_listings_count,availability_365)

# SELECT * FROM NY1

# WHERE Price > 0



##### Updating Category columns



# UPDATE NY_2

# SET [Price_Range_100] = CASE 



# 	WHEN price BETWEEN 0 AND 100 THEN '0-100'

# 	WHEN price BETWEEN 101 AND 200 THEN '101-200'

# 	WHEN price BETWEEN 201 AND 300 THEN '201-300'

# 	WHEN price BETWEEN 301 AND 400 THEN '301-400'

# 	WHEN price BETWEEN 401 AND 500 THEN '401-500'

# 	WHEN price BETWEEN 501 AND 600 THEN '501-600'

# 	WHEN price BETWEEN 601 AND 700 THEN '601-700'

# 	WHEN price BETWEEN 701 AND 800 THEN '701-800'

# 	WHEN price BETWEEN 801 AND 900 THEN '801-900'

# 	WHEN price BETWEEN 901 AND 1000 THEN '901-1000'

# 	ELSE '> 1000' END 

        

# , [Price_Range_Sequence] = CASE 



# 	WHEN price BETWEEN 0 AND 100 THEN 1

# 	WHEN price BETWEEN 101 AND 200 THEN 2

# 	WHEN price BETWEEN 201 AND 300 THEN 3

# 	WHEN price BETWEEN 301 AND 400 THEN 4

# 	WHEN price BETWEEN 401 AND 500 THEN 5

# 	WHEN price BETWEEN 501 AND 600 THEN 6

# 	WHEN price BETWEEN 601 AND 700 THEN 7

# 	WHEN price BETWEEN 701 AND 800 THEN 8

# 	WHEN price BETWEEN 801 AND 900 THEN 8

# 	WHEN price BETWEEN 901 AND 1000 THEN 10

# 	ELSE 11 END 

#         

# , [Availability_Range] =  CASE 

# 

# 	WHEN [availability_365] = 0 THEN 'No Availability'

# 	WHEN [availability_365] BETWEEN 1 AND 100 THEN 'Low Avalability'

# 	WHEN [availability_365] BETWEEN 101 AND 200 THEN 'Medium Availablity'

# 	WHEN [availability_365] BETWEEN 201 AND 300 THEN 'High Availablity'

# 	WHEN [availability_365] BETWEEN 301 AND 365 THEN 'Nearly Always Availability'

# 	ELSE 'Invalid' END

#        

# , [Availibility_Range_Sequence] = CASE 

# 

# 	WHEN [availability_365] = 0 THEN 0

# 	WHEN [availability_365] BETWEEN 1 AND 100 THEN 1

# 	WHEN [availability_365] BETWEEN 101 AND 200 THEN 2

# 	WHEN [availability_365] BETWEEN 201 AND 300 THEN 3

# 	WHEN [availability_365] BETWEEN 301 AND 365 THEN 4

# 	ELSE -1 END

#         

# 	FROM NY_2
### Lets Have a look at the Data we have prepared so far

data_cat=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_4.csv')

data_cat.head()
f,ax=plt.subplots(1,2,figsize=(20,10))

title = plt.title('Number of listings by Naighbourhood Groups', fontsize=20)

title.set_position([-0.2, 1.15])

patches, texts = plt.pie(data['neighbourhood_group'].value_counts(), startangle=90, pctdistance=0.85)

plt.legend(patches, data['neighbourhood_group'], loc="best")



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()



sns.countplot('neighbourhood_group',data=data,ax=ax[0])

ax[0].set_xlabel('Neighbourhood Groups')

ax[0].set_ylabel('Number of Listings')

plt.show()
# SELECT TOP 35 neighbourhood, count(*) FROM NY_2

# GROUP BY neighbourhood

# ORDER BY COUNT(*) DESC



#Resulted in following

data_neighbourhood=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_7.csv')

data_neighbourhood
#Lets run pie on this

plt.figure(figsize=(15,11))

plt.title('Number of listings by Top 35 Neighbourhoods', fontsize=20)

patches, texts = plt.pie(data_neighbourhood['COUNT'], startangle=90, pctdistance=0.85)

plt.legend(patches, data_neighbourhood['neighbourhood'], loc="upper right")



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()

plt.show()
# SELECT TOP 35 neighbourhood, count(*) AS [COUNT] FROM NY_2

# GROUP BY neighbourhood

# HAVING COUNT(*) > 500

# ORDER BY COUNT(*) DESC



#Resulted in following

data_neighbourhood_500=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_8.csv')

data_neighbourhood_500
f,ax=plt.subplots(1,2,figsize=(25,10))

title = plt.title('Number of listings by Naighbourhood having more than 500 records', fontsize=20)

title.set_position([0.0, 1.15])

patches, texts = plt.pie(data_neighbourhood_500['COUNT'], startangle=90, pctdistance=0.85)

plt.legend(patches, data_neighbourhood_500['neighbourhood'], loc="upper left")



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()



chart = sns.barplot(data=data_neighbourhood_500,x='neighbourhood',y='COUNT',ax=ax[0])

ax[0].set_xlabel('Neighbourhoods')

ax[0].set_ylabel('Number of Listings')

ax[0].tick_params(axis='x', labelrotation=45)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
plt.figure(figsize=(10,6))

title = plt.title('Number of listings by Room Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.countplot(x="room_type", data=data_cat)

ax.set_xlabel('Room Type')

ax.set_ylabel('Listings')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
plt.figure(figsize=(10,6))

title = plt.title('Number of Reviews by Room Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.barplot(x="room_type", y="number_of_reviews", data=data_cat, ci=None)

ax.set_xlabel('Room Type')

ax.set_ylabel('Reviews')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
plt.figure(figsize=(10,6))

title = plt.title('Distribution of Reviews by Neighbourhood Groups', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.violinplot(x="neighbourhood_group", y="number_of_reviews", data=data_cat)

ax.set_xlabel('Neighbourhood Group')

ax.set_ylabel('Reviews')

ax.set(ylim=(0, 200))

c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
plt.figure(figsize=(10,6))

title = plt.title('Distribution of Reviews by Room Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.violinplot(x="room_type", y="number_of_reviews", data=data_cat)

ax.set_xlabel('Room Types')

ax.set_ylabel('Reviews')

ax.set(ylim=(0, 200))

c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
heatmap_data_roomtype_ng_price = pd.pivot_table(data_cat, values='price', 

                     index=['room_type'], 

                     columns='neighbourhood_group')



plt.figure(figsize=(20,6))

title = plt.title('Room Type, Neighbourhood Groups and Price Relation', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data_roomtype_ng_price, cmap="YlGnBu", cbar_kws={'label': 'Price'})

ax.set_xlabel('Neighbourhood Groups')

ax.set_ylabel('Room Type')

a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
#SELECT Price_Range_100, COUNT(*) AS [COUNT]

#FROM NY_2

#GROUP BY Price_Range_100, Price_Range_Sequence

#ORDER BY Price_Range_Sequence

data_price_count=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_5.csv')

data_price_count
#SELECT [Availability_Range], COUNT(*) AS [COUNT]

#FROM NY_2

#GROUP BY [Availability_Range], [Availibility_Range_Sequence]

#ORDER BY [Availibility_Range_Sequence]

data_availability_count=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_6.csv')

data_availability_count
f,ax=plt.subplots(1,2,figsize=(20,10))

plt.title('Share of listings by Availability', fontsize=20)

wedges, patches, texts = ax[1].pie(data_availability_count['COUNT'], startangle=90, labels=data_availability_count['Availability_Range'], pctdistance=0.6,autopct='%1.1f%%')



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()



chart = sns.barplot(data=data_availability_count,x='Availability_Range',y='COUNT',ax=ax[0])

ax[0].set_title('Share of listings by Availability', fontsize=20)

ax[0].set_xlabel('Availability')

ax[0].set_ylabel('Number of Listings')

chart.set_xticklabels(chart.get_xticklabels(), rotation=0, horizontalalignment='center')

plt.show()
data_availability_count_nonzero=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_9.csv')

f,ax=plt.subplots(1,2,figsize=(20,10))

plt.title('Share of listings by Availability with non zero availability', fontsize=20)

wedges, patches, texts = plt.pie(data_availability_count_nonzero['COUNT'], startangle=90, labels=data_availability_count_nonzero['Availability_Range'], pctdistance=0.6,autopct='%1.1f%%')



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()



chart = sns.barplot(data=data_availability_count_nonzero,x='Availability_Range',y='COUNT',ax=ax[0])

ax[0].set_title('Share of listings by Availability with non zero availability', fontsize=20)

ax[0].set_xlabel('Availability')

ax[0].set_ylabel('Number of Listings')

chart.set_xticklabels(chart.get_xticklabels(), rotation=0, horizontalalignment='center')
#Including more Parameters

plt.figure(figsize=(10,10))

title = plt.title('Number of listings by Availability and Neighbourhood Groups', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.countplot(x="Availability_Range", data=data_cat, hue='neighbourhood_group')

ax.set_xlabel('Availability')

ax.set_ylabel('Number of Listings')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')

ax.titlesize = 'large'   
#### Removing Zero Availability



# Removing records with zero availability from *data_cat* and loading into *data_cat_nonzero* by following __SQL__



# SELECT * FROM NY_2

# WHERE availability_365 > 0

# Resulted in 31354 records



data_cat_nonzero=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_10.csv')
plt.figure(figsize=(10,6))

title = plt.title('Number of listings by Availability and Neighbourhood Groups', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.countplot(x="Availability_Range", data=data_cat_nonzero, hue='neighbourhood_group')

ax.set_xlabel('Availability')

ax.set_ylabel('Number of Listings')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
plt.figure(figsize=(10,6))

title = plt.title('Number of listings by Availability and Room Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.countplot(x="Availability_Range", data=data_cat_nonzero, hue='room_type')

ax.set_xlabel('Availability')

ax.set_ylabel('Number of Listings')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
f,ax=plt.subplots(1,1,figsize=(20,10))



chart1 = sns.barplot(x="Availability_Range", y="number_of_reviews", data=data_cat, ci=None)

title = ax.set_title('Number of Reviews by Availability', fontsize=20)

title.set_position([0.5, 1.15])

ax.set_xlabel('Availability')

ax.set_ylabel('Number Reviews')

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=0, horizontalalignment='center')



plt.show()
plt.figure(figsize=(10,6))

title = plt.title('Number of Reviews by Availability and Appartment Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.barplot(x="Availability_Range", y="number_of_reviews", data=data_cat_nonzero, hue='room_type', ci=None)

ax.set_xlabel('Availability')

ax.set_ylabel('Reviews')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
#Another incoming from SQL

# SELECT Availibility_Range_Sequence, neighbourhood_group, price

# FROM NY_2

# ORDER BY Availibility_Range_Sequence



data_av_heatmap=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_11.csv')

heatmap_data = pd.pivot_table(data_av_heatmap, values='price', 

                     index=['Availibility_Range_Sequence'], 

                     columns='neighbourhood_group')



plt.figure(figsize=(10,6))

title = plt.title('Availability, Neighbour Group, and Price comparison', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})

ax.set_xlabel('Neighbourhood Groups')

ax.set_ylabel('Availability')

y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']

a = ax.set_yticklabels(y_label_list, rotation=0)

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
# Heatmap with respect to Top 10 Neighbourhoods

# SELECT Availibility_Range_Sequence, neighbourhood, price FROM NY_2

# WHERE neighbourhood IN

# (

# 	SELECT TOP 10 neighbourhood

# 	FROM NY_2

# 	GROUP BY neighbourhood

# 	ORDER BY COUNT(*) DESC

#)



data_av_heatmap=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_12.csv')

heatmap_data = pd.pivot_table(data_av_heatmap, values='price', 

                     index=['Availibility_Range_Sequence'], 

                     columns='neighbourhood')



plt.figure(figsize=(20,6))

title = plt.title('Availability, Top 10 Neighbourhoods, and Price comparison', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})

ax.set_xlabel('Neighbourhoods')

ax.set_ylabel('Availability')

y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']

a = ax.set_yticklabels(y_label_list, rotation=0)

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
# SELECT Availibility_Range_Sequence, room_type, price

# FROM NY_2

# ORDER BY Availibility_Range_Sequence



data_av_heatmap_3=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_13.csv')

heatmap_data = pd.pivot_table(data_av_heatmap_3, values='price', 

                     index=['Availibility_Range_Sequence'], 

                     columns='room_type')



plt.figure(figsize=(10,6))

title = plt.title('Availability, Room Type, and Price comparison', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})

ax.set_xlabel('Room Type')

ax.set_ylabel('Availability')

y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']

a = ax.set_yticklabels(y_label_list, rotation=0)

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
f,ax=plt.subplots(1,2,figsize=(15,10))

title = plt.title('Number of listings by Price', fontsize=20)

title.set_position([0.0, 1.15])

patches, texts = plt.pie(data_price_count['COUNT'], startangle=90, pctdistance=0.6)

plt.legend(patches, data_price_count['Price_Range_100'], loc="upper right")

ax[1].set_title('Number of listings by Price', fontsize=20)



centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.tight_layout()



chart = sns.barplot(data=data_price_count,x='Price_Range_100',y='COUNT',ax=ax[0])

ax[0].set_xlabel('Price')

ax[0].set_ylabel('Number of Listings')

chart.set_xticklabels(chart.get_xticklabels(), rotation=35, horizontalalignment='right')

plt.show()
plt.figure(figsize=(10,6))

title = plt.title('Price comparison by Room Type', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.violinplot(x="room_type", y="price", data=data_cat_nonzero)

ax.set_xlabel('Room Type')

ax.set_ylabel('Price')

ax.set(ylim=(0, 600))

c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
plt.figure(figsize=(10,6))

title = plt.title('Price comparison by Neighbourhood Groups', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.violinplot(x="neighbourhood_group", y="price", data=data_cat_nonzero)

ax.set_xlabel('Neighbourhood Groups')

ax.set_ylabel('Price')

ax.set(ylim=(0, 600))

c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
# SELECT * FROM NY_2

# WHERE neighbourhood IN

# (

# 	SELECT TOP 20 neighbourhood

# 	FROM NY_2

# 	WHERE availability_365 > 0

# 	GROUP BY neighbourhood

# 	ORDER BY COUNT(*) DESC

# )

# AND availability_365 > 0

data_top_10_neighbourhoods=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_14.csv')



plt.figure(figsize=(25,6))

plt.title('Price comparison by Top 10 Neighbourhoods', fontsize=20)

ax = sns.violinplot(x="neighbourhood", y="price", data=data_top_10_neighbourhoods)

ax.set_xlabel('Neighbourhoods')

ax.set_ylabel('Price')

ax.set(ylim=(0, 500))

c = ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax = sns.jointplot(y="price", x="availability_365", data=data_cat_nonzero,xlim=(0, 365), ylim=(0, 500), height=8, ratio=5)

l = ax.set_axis_labels("Availaility", "Price")

plt.subplots_adjust(top=0.9)

plt.suptitle('Price and availability distribution', fontsize = 16)

plt.show()
heatmap_data_ng_price_reviews = pd.pivot_table(data_cat_nonzero, values='number_of_reviews', 

                     index=['neighbourhood_group'], 

                     columns='Price_Range_100')



plt.figure(figsize=(20,6))

title = plt.title('Reviews by Neighbourhood and Price Range', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data_ng_price_reviews, cmap="YlGnBu", cbar_kws={'label': 'Reviews'})

ax.set_xlabel('Price Range')

ax.set_ylabel('Neighbourhood Groups')

a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
heatmap_data_rt_price_reviews = pd.pivot_table(data_cat_nonzero, values='number_of_reviews', 

                     index=['room_type'], 

                     columns='Price_Range_100')



plt.figure(figsize=(20,6))

title = plt.title('Reviews by Room Type and Price Range', fontsize=20)

title.set_position([0.5, 1.15])

ax = sns.heatmap(heatmap_data_rt_price_reviews, cmap="YlGnBu", cbar_kws={'label': 'Reviews'})

ax.set_xlabel('Price Range')

ax.set_ylabel('Room Type')

a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')

a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
import urllib

f,ax=plt.subplots(1,2,figsize=(25,10))

nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

ax[0].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

map1 = sns.scatterplot(data_cat.longitude,data_cat.latitude,hue=data_cat.neighbourhood_group, ax=ax[0])

title = ax[0].set_title('Data with 0 Availability', fontsize=20)

title.set_position([0.5, 1.1])



ax[1].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

map2 = sns.scatterplot(data_cat_nonzero.longitude,data_cat_nonzero.latitude,hue=data_cat_nonzero.neighbourhood_group, ax=ax[1])

title = ax[1].set_title('Data without 0 Availability', fontsize=20)

title.set_position([0.5, 1.1])

plt.show()
ax=plt.figure(figsize=(20,8))

nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()

sns.scatterplot(data=data_cat[data_cat['Availability_Range']=='No Availability'], x='longitude',y='latitude',hue='Availability_Range', ax=ax)

title = ax.set_title('Data for No Availability', fontsize=20)

title.set_position([0.5, 1.1])
f,ax=plt.subplots(1,2,figsize=(25,10))

nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

ax[0].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])



map1 = sns.scatterplot(data = data_cat_nonzero,x='longitude',y='latitude',hue='Price_Range_100', ax=ax[0], palette='RdBu')

title = ax[0].set_title('Distribution by Price', fontsize=20)

title.set_position([0.5, 1.1])



ax[1].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

map2 = sns.scatterplot(data = data_cat_nonzero,x='longitude',y='latitude',hue='Availability_Range', ax=ax[1])

title = ax[1].set_title('Distribution by Availability', fontsize=20)

title.set_position([0.5, 1.1])

plt.show()
ax=plt.figure(figsize=(20,8))

nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()

sns.scatterplot(data=data_cat_nonzero, x='longitude',y='latitude',hue='room_type', ax=ax)

title = ax.set_title('Distribution by Room Type', fontsize=20)

title.set_position([0.5, 1.1])