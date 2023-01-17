#Importing libraries

import pandas as pd

import seaborn as sns

import numpy as np

import folium as fo

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
#reading data

df=pd.read_csv('/kaggle/input/montcoalert/911.csv')
# A quick loock in the data

df.head()
df.info()
#Creating columns that might be useful

df['timeStamp']=df['timeStamp'].apply(pd.Timestamp)

df['Year']=df.timeStamp.dt.year

df['Month']=df.timeStamp.dt.month
df['Year'].value_counts()
#since 2015 and 2020 are not complete, we're not using them

df=df[df['Year'].isin([2016,2017,2018,2019])]
df['title'].unique()
#There are 3 major categories: EMS (Emergency Medical Services), Traffic and Fire, 

#so let's use them to aggregate data

df['title']=df['title'].str.split(':').str.get(0)
#How many calls per year?

plt.figure(figsize=(10,6))

sns.set_palette('tab10')

sns.set_style('darkgrid')

sns.countplot(x='Year',data=df)

plt.title('Number of calls per year')
#How many calls per month over the years?

plt.figure(figsize=(13,7))

sns.set_style('darkgrid')

sns.countplot(x='Month',hue='Year',data=df)

plt.title('Number of calls per month over the years')
#How many calls per category (title)?

plt.figure(figsize=(10,6))

sns.set_style('darkgrid')

sns.countplot(x='Year',hue='title',data=df,hue_order=['EMS','Traffic','Fire'])

plt.title('Number of calls per per category over the years')
#How many calls per category (title) over the months?

plt.figure(figsize=(10,6))

sns.set_style('darkgrid')

sns.countplot(x='Month',hue='title',data=df,hue_order=['EMS','Traffic','Fire'])

plt.title('Number of calls per per category over the months')
EMS=df[df['title']=='EMS']
#Top 25 towns with more calls

top_25=EMS['twp'].value_counts(ascending=False,normalize=True).head(25).index

plt.figure(figsize=(10,6))

sns.countplot(y='twp',data=EMS,order=top_25)

plt.title('Top 25 towns with more calls')
df.head(3)
#Let's plot a few points to take a look

Map=fo.Map([40.121354,-75.363829],zoom_start=7)

random_index=np.random.choice(df.index,1000) #getting some random points to plot

for ind in random_index:

    lat=df.loc[ind,'lat']

    long=df.loc[ind,'lng']

    fo.CircleMarker([lat,long],radius=2).add_to(Map)

Map
sns.scatterplot(x=EMS['lat'],y=EMS['lng'],data=df, alpha=0.3)
#Analysing Latitude

sns.boxplot(x=EMS['lat'])
#Getting outliers data (take a look at https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02)

Q1=EMS['lat'].quantile(.25)

Q3=EMS['lat'].quantile(.75)

IQR=Q3-Q1

Lower_Whisker=Q1-1.5*IQR

Upper_Whisker = Q3+1.5*IQR
#How many outliers?

EMS[EMS['lat']>Upper_Whisker].shape[0]+EMS[EMS['lat']<Lower_Whisker].shape[0]
#Let's save the outliers for posteriori analysis

outliers=EMS[(EMS['lat']>Upper_Whisker)|(EMS['lat']<Lower_Whisker)]
#removing outliers

EMS_treated=EMS[(EMS['lat']<Upper_Whisker)&(EMS['lat']>Lower_Whisker)]
#Plotting the treated data 

sns.scatterplot(x=EMS_treated['lat'],y=EMS_treated['lng'],data=df)
#boxplot with treated data

sns.boxplot(x=EMS_treated['lat'])
#the same process for long

sns.boxplot(x=EMS_treated['lng'])
Q1=EMS_treated['lng'].quantile(.25)

Q3=EMS_treated['lng'].quantile(.75)

IQR=Q3-Q1

Lower_Whisker=Q1-1.5*IQR

Upper_Whisker = Q3+1.5*IQR
#How many outliers?

EMS_treated[EMS_treated['lng']>Upper_Whisker].shape[0]+EMS_treated[EMS_treated['lng']<Lower_Whisker].shape[0]
#saving outliers

outliers=pd.concat([EMS_treated[(EMS_treated['lng']>Upper_Whisker)|(EMS_treated['lng']<Lower_Whisker)],outliers])
#cleaning longitude outliers

EMS_treated=EMS_treated[(EMS_treated['lng']<Upper_Whisker)&(EMS_treated['lng']>Lower_Whisker)]
#Total outliers

outliers.shape[0]
#Percentagem in relation to the whole data

(outliers.shape[0]/EMS.shape[0])*100
#Let's see again the treated data

#It's clear the improvement of this plot in relation to the first ones

sns.scatterplot(x=EMS_treated['lat'],y=EMS_treated['lng'],data=df)
sns.boxplot(x=EMS_treated['lng'])
outliers.head(1)
#Outliers in red

Map_outliers=fo.Map([40.269061,-75.69959],zoom_start=6)

for index,row in outliers.iterrows():

    lat=row['lat']

    long=row['lng']

    fo.CircleMarker([lat,long],radius=2,color='red').add_to(Map_outliers)



# Treated data in blue

random_indexes=np.random.choice(EMS_treated.index,2000)

for rand_in in random_indexes:

    lat=EMS_treated.loc[rand_in,'lat']

    long=EMS_treated.loc[rand_in,'lng']

    fo.CircleMarker([lat,long],radius=2,color='blue').add_to(Map_outliers)

Map_outliers
#Getting lat long

X=np.array(EMS_treated[['lat','lng']])
#creating instance of KMeans

kmeans=KMeans(init='k-means++')
#Creating instance of Elbow Visualizer to find how many clusters we use

visualizer = KElbowVisualizer(model=kmeans, k=(5,20))
#Fitting the model

visualizer.fit(X)      
clustering=KMeans(init='k-means++',n_clusters=10)

clustering.fit(X)    
clusters=clustering.cluster_centers_
Map=fo.Map([ 40.13572425, -75.20909773],zoom_start=8)

for i in range(1000):

    random_index=np.random.choice(EMS_treated.index,1)

    lat=df.loc[random_index,'lat']

    long=df.loc[random_index,'lng']

    fo.Circle([lat,long],radius=2).add_to(Map)     

    

for c in clusters:

    lat=c[0]

    long=c[1]

    fo.RegularPolygonMarker([lat,long],radius=4,number_of_sides=3,color='black').add_to(Map) 

Map
#Clustering including outliers

X_outliers=np.array(EMS[['lat','lng']])

clustering_outliers=KMeans(init='k-means++',n_clusters=10)

clustering_outliers.fit(X_outliers)

clusters_outliers=clustering_outliers.cluster_centers_
#Adding the outlier clusters to the Map

for c in clusters_outliers:

    lat=c[0]

    long=c[1]

    fo.RegularPolygonMarker([lat,long],radius=4,number_of_sides=3,color='red').add_to(Map) 

Map