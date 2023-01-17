# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns# for data viz.

import geopandas

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nyc_airbnb=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')# accessing the csv file
print(nyc_airbnb.shape)

nyc_airbnb.head(8)
#checking DataType of every column in the dataset

nyc_airbnb.dtypes

numeric_nyc_airbnb=nyc_airbnb._get_numeric_data().columns# to Extract the names of columns that are Numeric



cat_nyc_airbnb=set(nyc_airbnb.columns)-set(numeric_nyc_airbnb) # To Extract the names of columns that are Categorical

# to findout number of null values in Each column and suming their count

nyc_airbnb.isnull().sum()
num_nyc_airbnb=nyc_airbnb._get_numeric_data()

#print(num_nyc_airbnb.head())

num_nyc_airbnb.replace([np.inf, -np.inf], np.nan).dropna(axis=1)# replaced all values of Infinity with Nan



# print(num_nyc_airbnb.dtypes)# checking datatypes of  numeric columns



# print("------------------------------------------------")

# print(num_nyc_airbnb.isnull().sum()) # This shows that oly reviews_per_month had missing values so we have to fill it up



#replacing all NaN values in 'reviews_per_month' with 0

num_nyc_airbnb.fillna({'reviews_per_month':0}, inplace=True)

#examing changes

# print("------------------------------------------------")

# print(num_nyc_airbnb.reviews_per_month.isnull().sum()) # so we have removed all the Nan Values

plt.figure(figsize=(10,10))

ax = sns.countplot(nyc_airbnb["neighbourhood_group"])
plt.figure(figsize=(10,10))

ax = sns.countplot(nyc_airbnb['neighbourhood_group'],hue=nyc_airbnb['room_type'])

# here hue parameter will give us 3 diffrent colors
#Brooklyn

sub_1=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Brooklyn') & (nyc_airbnb['room_type']=='Shared room')]

price_sub1=num_nyc_airbnb['price'].iloc[sub_1.index]# prices for Neighbourhood group having Brooklyn





#Bronx

sub_2=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Bronx') & (nyc_airbnb['room_type']=='Shared room')]

price_sub2=num_nyc_airbnb['price'].iloc[sub_2.index]# prices for Neighbourhood group having Bronx



#Staten Island

sub_3=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Staten Island') & (nyc_airbnb['room_type']=='Shared room')]

price_sub3=num_nyc_airbnb['price'].iloc[sub_3.index]# prices for Neighbourhood group having Staten Island



#Queens

sub_4=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Queens') & (nyc_airbnb['room_type']=='Shared room')]

price_sub4=num_nyc_airbnb['price'].iloc[sub_4.index]# prices for Neighbourhood group having Queens





#Manhattan

sub_5=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Manhattan') & (nyc_airbnb['room_type']=='Shared room')]

price_sub5=num_nyc_airbnb['price'].iloc[sub_5.index]# prices for Neighbourhood group having Manhattan





percentile_price_brooklyn=[]#percentile of prices for Private rooms near brooklyn



percentile_price_Bronx=[]#percentile of prices for Private rooms near Bronx



percentile_price_Staten_Island=[] #percentile of prices for Private rooms near Staten Island



percentile_price_Queens=[] #percentile of prices for accomodations near Queens



percentile_price_Manhattan=[] #percentile of prices for accomodations near Manhattan





percentiles=[]# percentiles

for i in range(25,91):

    percentile_price_brooklyn.append(int(price_sub1.quantile(i/100)))

    percentile_price_Bronx.append(int(price_sub2.quantile(i/100)))

    percentile_price_Staten_Island.append(int(price_sub3.quantile(i/100)))

    percentile_price_Queens.append(int(price_sub4.quantile(i/100)))

    percentile_price_Manhattan.append(int(price_sub5.quantile(i/100)))

    percentiles.append(i)

    



plt.title('Prices of 90% Shared Rooms',fontsize=15,color='Red')

sns.set_style("darkgrid")



# for i in range(2):



sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_brooklyn),label='Brooklyn')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Bronx),label='Bronx')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Staten_Island),label='Staten Island')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Queens),label='Queens')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Manhattan),label='Manhattan')



sd.set(xlabel='Percentiles', ylabel='Percentile Prices in U.S $')



#Brooklyn

sub_1=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Brooklyn') & (nyc_airbnb['room_type']=='Private room')]

price_sub1=num_nyc_airbnb['price'].iloc[sub_1.index]# prices for Neighbourhood group having Brooklyn





#Bronx

sub_2=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Bronx') & (nyc_airbnb['room_type']=='Private room')]

price_sub2=num_nyc_airbnb['price'].iloc[sub_2.index]# prices for Neighbourhood group having Bronx



#Staten Island

sub_3=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Staten Island') & (nyc_airbnb['room_type']=='Private room')]

price_sub3=num_nyc_airbnb['price'].iloc[sub_3.index]# prices for Neighbourhood group having Staten Island



#Queens

sub_4=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Queens') & (nyc_airbnb['room_type']=='Private room')]

price_sub4=num_nyc_airbnb['price'].iloc[sub_4.index]# prices for Neighbourhood group having Queens





#Manhattan

sub_5=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Manhattan') & (nyc_airbnb['room_type']=='Private room')]

price_sub5=num_nyc_airbnb['price'].iloc[sub_5.index]# prices for Neighbourhood group having Manhattan





percentile_price_brooklyn=[]#percentile of prices for Private rooms near brooklyn



percentile_price_Bronx=[]#percentile of prices for Private rooms near Bronx



percentile_price_Staten_Island=[] #percentile of prices for Private rooms near Staten Island



percentile_price_Queens=[] #percentile of prices for accomodations near Queens



percentile_price_Manhattan=[] #percentile of prices for accomodations near Manhattan





percentiles=[]# percentiles

for i in range(25,91):

    percentile_price_brooklyn.append(int(price_sub1.quantile(i/100)))

    percentile_price_Bronx.append(int(price_sub2.quantile(i/100)))

    percentile_price_Staten_Island.append(int(price_sub3.quantile(i/100)))

    percentile_price_Queens.append(int(price_sub4.quantile(i/100)))

    percentile_price_Manhattan.append(int(price_sub5.quantile(i/100)))

    percentiles.append(i)

    



plt.title('Prices of 90% Private Rooms',fontsize=15,color='Red')

sns.set_style("darkgrid")



sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_brooklyn),label='Brooklyn')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Bronx),label='Bronx')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Staten_Island),label='Staten Island')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Queens),label='Queens')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Manhattan),label='Manhattan')



sd.set(xlabel='Percentiles', ylabel='Percentile Prices in U.S $')



room_type='Entire home/apt'

#Brooklyn

sub_1=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Brooklyn') & (nyc_airbnb['room_type']==room_type)]

price_sub1=num_nyc_airbnb['price'].iloc[sub_1.index]# prices for Neighbourhood group having Brooklyn





#Bronx

sub_2=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Bronx') & (nyc_airbnb['room_type']==room_type)]

price_sub2=num_nyc_airbnb['price'].iloc[sub_2.index]# prices for Neighbourhood group having Bronx



#Staten Island

sub_3=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Staten Island') & (nyc_airbnb['room_type']==room_type)]

price_sub3=num_nyc_airbnb['price'].iloc[sub_3.index]# prices for Neighbourhood group having Staten Island



#Queens

sub_4=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Queens') & (nyc_airbnb['room_type']==room_type)]

price_sub4=num_nyc_airbnb['price'].iloc[sub_4.index]# prices for Neighbourhood group having Queens





#Manhattan

sub_5=nyc_airbnb.loc[(nyc_airbnb['neighbourhood_group'] == 'Manhattan') & (nyc_airbnb['room_type']==room_type)]

price_sub5=num_nyc_airbnb['price'].iloc[sub_5.index]# prices for Neighbourhood group having Manhattan





percentile_price_brooklyn=[]#percentile of prices for Private rooms near brooklyn



percentile_price_Bronx=[]#percentile of prices for Private rooms near Bronx



percentile_price_Staten_Island=[] #percentile of prices for Private rooms near Staten Island



percentile_price_Queens=[] #percentile of prices for accomodations near Queens



percentile_price_Manhattan=[] #percentile of prices for accomodations near Manhattan





percentiles=[]# percentiles

for i in range(25,91):

    percentile_price_brooklyn.append(int(price_sub1.quantile(i/100)))

    percentile_price_Bronx.append(int(price_sub2.quantile(i/100)))

    percentile_price_Staten_Island.append(int(price_sub3.quantile(i/100)))

    percentile_price_Queens.append(int(price_sub4.quantile(i/100)))

    percentile_price_Manhattan.append(int(price_sub5.quantile(i/100)))

    percentiles.append(i)

    



plt.title('Prices of 90% Entire home/apt',fontsize=15,color='Red')

sns.set_style("darkgrid")



sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_brooklyn),label='Brooklyn')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Bronx),label='Bronx')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Staten_Island),label='Staten Island')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Queens),label='Queens')

sd=sns.lineplot(x=pd.Series(percentiles),y=pd.Series(percentile_price_Manhattan),label='Manhattan')



sd.set(xlabel='Percentiles', ylabel='Percentile Prices in U.S $')





# f, axes = plt.subplots(2,1,figsize = (10,10))

# sns.set_style("dark")

# # for i in range(2):

# sns.lineplot(x=pd.Series(percentiles_brooklyn),y=pd.Series(percentile_price_brooklyn),ax=axes[0])

# sns.lineplot(x=pd.Series(percentiles_Bronx),y=pd.Series(percentile_price_Bronx),ax=axes[1])

crs = {'init':'epsg:4326'}

geometry = geopandas.points_from_xy(nyc_airbnb.longitude, nyc_airbnb.latitude)

geo_data = geopandas.GeoDataFrame(nyc_airbnb,crs=crs,geometry=geometry)
nyc = geopandas.read_file(geopandas.datasets.get_path('nybb'))

nyc = nyc.to_crs(epsg=4326)
fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='availability_365',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Number of days when listing is available for booking")

plt.axis('off')
