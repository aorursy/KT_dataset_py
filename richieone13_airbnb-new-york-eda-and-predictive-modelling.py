import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

from shapely import wkt
# Reading the files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#Make the data look cleaner, to 1.dp when finding out statistical analysis on the price

pd.set_option("display.precision", 1)
df.head()
#check the amount of rows and columns within the dataset

df.shape
#Checking for Null values in the dataset

print('Null values in Airbnb dataset: \n')

print(df.isnull().sum())

print('\n')

print('Percentage of null values in review columns: ')

print(round(df['last_review'].isnull().sum()/len(df)*100, 2),"%")
#Review the listings by boroname

plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=df, palette="coolwarm")
#let's proceed examine the categorical unique values



#examining the unique values of group

print('There are X number of unique neighbourhoods:')

print(len(df.neighbourhood_group.unique()))

print('\n')

print(df.neighbourhood_group.unique())
print('There are X number of unique neighbourhoods:')

print(len(df.neighbourhood.unique()))

# print('\n')

# print(df.neighbourhood.unique())
print('There are X number of unique room types:')

print(len(df.room_type.unique()))

print('\n')

print(df.room_type.unique())
#Here we are retrieve the NYC boroughs from Geopandas

nyc = gpd.read_file(gpd.datasets.get_path('nybb'))

nyc.head(5)
#Get a count by borough

borough_count = df.groupby('neighbourhood_group').agg('count').reset_index()



#Rename the column to boroname, so that we can join the data to it on a common field

nyc.rename(columns={'BoroName':'neighbourhood_group'}, inplace=True)

nyc_geo = nyc.merge(borough_count, on='neighbourhood_group')
#Plot the count by borough into a map

fig,ax = plt.subplots(1,1, figsize=(20,10))

nyc_geo.plot(column='id', cmap='coolwarm', alpha=.5, ax=ax, legend=True)

nyc_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)

plt.title("Number of Airbnb Listings by NYC Borough")

plt.axis('off')



#Thanks to @geowiz34 https://www.kaggle.com/geowiz34 for this plot.
import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)



from folium.plugins import FastMarkerCluster



Lat=40.80

Long=-73.80



locations = list(zip(df.latitude, df.longitude))



map1 = folium.Map(location=[Lat,Long], zoom_start=11)

FastMarkerCluster(data=locations).add_to(map1)

map1
plt.figure(figsize=(15,6))

sns.countplot(data=df, x='neighbourhood_group', hue='room_type', palette='coolwarm')

plt.title('Counts of AirBnB in neighbourhood group with Room Type Category', fontsize=15)

plt.xlabel('Neighbourhood group')

plt.ylabel("Count")

plt.legend(frameon=False, fontsize=12)
df.pivot_table(index=['neighbourhood_group'],

               values='price',

               aggfunc=['count', 'mean','median', 'std'])
#Alternative to than using pivot tables

df.groupby('neighbourhood_group').count()['price'].nlargest(n=20, keep='all')
plt.figure(figsize=(15,6))

sns.violinplot(data=df[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')

plt.title('Density and distribution of prices - ALL TYPES OF ROOMS', fontsize=15)

plt.xlabel('Neighbourhood group')

plt.ylabel("Price")
dfa = df[df.room_type== 'Entire home/apt'] # panda chain rule

print(dfa.shape)
plt.figure(figsize=(15,6))

sns.violinplot(data=dfa[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')

plt.title('Density and distribution of prices - ENITRE APP', fontsize=15)

plt.xlabel('Neighbourhood group')

plt.ylabel("Price")
dfp = df[df.room_type== 'Private room'] # panda chain rule



plt.figure(figsize=(15,6))

sns.violinplot(data=dfp[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')

plt.title('Density and distribution of prices - PRIVATE ROOM', fontsize=15)

plt.xlabel('Neighbourhood group')

plt.ylabel("Price")
#Top 10 neighbourhoods

df.groupby('neighbourhood').mean()['price'].nlargest(n=10, keep='all')
#word cloud visualisation to show the popular neighbourhoods



from wordcloud import WordCloud



plt.subplots(figsize=(20,15))

wordcloud = WordCloud(

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.neighbourhood))

plt.imshow(wordcloud)

plt.title('Word Cloud for Neighbourhoods')

plt.axis('off')

plt.show()
airbnb=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)

airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

#examing the changes

airbnb.head(5)

#Encode the input Variables

def Encode(airbnb):

    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'room_type'])]:

        airbnb[column] = airbnb[column].factorize()[0]

    return airbnb



airbnb_en = Encode(airbnb.copy())
airbnb_en.head(15)

#Get Correlation between different variables

corr = airbnb_en.corr(method='kendall')

plt.figure(figsize=(14,9))

sns.heatmap(corr, annot=True)

airbnb_en.columns
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score
#Defining the independent variables and dependent variables

x = airbnb_en.iloc[:,[0,1,3,4,5]]

y = airbnb_en['price']

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)

x_train.head()

y_train.head()
x_train.shape
#Prepare a Linear Regression Model

reg=LinearRegression()

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
#Prepairng a Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_predict)