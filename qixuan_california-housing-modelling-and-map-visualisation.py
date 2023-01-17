# This Python 3 environment comes with many helpful analytics libraries installed

#importing the libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import os
#loading the dataset and obtaining info about columns

df_house=pd.read_csv("../input/california-housing-prices/housing.csv")

df_house.head()
df_house.shape
df_house.describe()
df_house.isnull().sum()
df_house[df_house.isna().any(axis=1)]
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)

df_train=df_house.drop(['ocean_proximity'],axis=1)

imp.fit(df_train.values) 

X_test = df_house

result=imp.transform(df_train.values)

df_house1=pd.DataFrame(result, columns=['longitude','latitude','housing_median_age','total_rooms',

                                       'total_bedrooms','population','households','median_income',

                                       'median_house_value']) 

df_house1.head()
df_house1['ocean_proximity']=df_house['ocean_proximity']

df_house1.head()
df_house1.isnull().sum()
df_house1.describe(include=['O'])
df_house1.ocean_proximity.unique()
df_house1.ocean_proximity.value_counts().plot(kind="bar")

plt.title('Number of houses')

plt.xlabel("Ocean proximity")

plt.ylabel('Count')

plt.show()
df_house1.ocean_proximity.value_counts()
df_house1['median_house_value'].describe()
df_house1.boxplot(column='median_house_value',sym='k.')

plt.show()
df_house1['log_median_house_value']=np.log(df_house1.median_house_value)

#Now plot a boxplot again

df_house1.boxplot(column='log_median_house_value',sym='k.')

plt.show()
#draw boxplot

df_house1.boxplot(column="median_house_value", by='ocean_proximity', sym = 'k.', figsize=(18,6))

#set title

plt.title('Boxplot for Camparing price per living space for each city')

plt.show()
plt.figure(figsize=(10,8))

corr = df_house1.corr()

#use seaborn to draw the headmap

sns.heatmap(corr, 

            xticklabels=corr.columns.values, #x label

            yticklabels=corr.columns.values) #y label

plt.show()
#plot a price histgram

df_house1['median_house_value'].hist(bins=100)

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=df_house1.latitude.values,y=df_house.longitude.values,size=10)

plt.ylabel("longitude")

plt.xlabel("latitude")

plt.show()

sns.despine
plt.scatter(df_house1.median_house_value,df_house.median_income)

plt.show()
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

from sklearn.model_selection import train_test_split

labels=df_house1['median_house_value']

train1=df_house1.drop(['median_house_value','log_median_house_value','ocean_proximity'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(train1, labels, test_size=0.10, random_state=2)

reg.fit(x_train,y_train)
reg.score(x_test,y_test)
#Gradient descent boosting

from sklearn.ensemble import GradientBoostingRegressor

clf=GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1,loss="ls")

clf.fit(x_train,y_train)
clf.score(x_test, y_test)
from mpl_toolkits.basemap import Basemap



m = Basemap(projection='mill',llcrnrlat=25,urcrnrlat=49.5,\

            llcrnrlon=-140,urcrnrlon=-50,resolution='l')



plt.figure(figsize=(25,17))

m.drawcountries() 

m.drawstates()  

m.drawcoastlines()

x,y = m(-119.4179,36.7783)

m.plot(x, y, 'ro', markersize=20, alpha=.8) 

m.bluemarble() 

m.drawmapboundary(color = '#FFFFFF')

plt.show()
import folium

from folium.plugins import HeatMap

map_hooray = folium.Map(location=[36.7783,-119.4179],

                    zoom_start = 6, min_zoom=5) 



df = df_house1[['latitude', 'longitude']]

data = [[row['latitude'],row['longitude']] for index, row in df.iterrows()]

HeatMap(data, radius=10).add_to(map_hooray)

map_hooray