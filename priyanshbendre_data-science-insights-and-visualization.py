import pandas as pd

import numpy as np

import seaborn as sns

%pylab inline

import sklearn as sk

import sklearn.tree as tree

from IPython.display import Image  

import pydotplus

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')

df.set_index('ID',inplace=True)
df.head(10)
len(df)
df['Primary Type'].value_counts().head(5)
(df['Primary Type'].value_counts()/len(df)).head(5)
df2=df.copy()

df2.drop(columns=['Unnamed: 0','Case Number','Block','IUCR','FBI Code','Updated On','X Coordinate','Beat','District',\

                  'Community Area','Y Coordinate','Location'],inplace=True)
df2 = df2[~(df2.Year == 2017)]
df2=df2.loc[~df2['Latitude'].isnull()]
df2['Arrest']=(df.Arrest==True)*1

df2['Domestic']=(df.Domestic==True)*1
df2['month']=df2.Date.astype(np.datetime64).dt.month
df2.groupby('Ward')['Arrest'].sum().nlargest(1)
lat_mean=df2[df2.Ward==28].Latitude.mean()

long_mean=df2[df2.Ward==28].Longitude.mean()

location_mean=[lat_mean,long_mean]

location_mean
from haversine import *

def haversine_np(lon1, lat1):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, long_mean, lat_mean])



    distance_lon = lon2 - lon1

    distance_lat = lat2 - lat1



    a = np.sin(distance_lat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(distance_lon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
df2['distance']=df2.apply(lambda x: haversine_np(x.Longitude,x.Latitude),axis=1)
df2['crime_year']=df2.groupby('Year')['Arrest'].transform('size')

df2['crime_year_ward']=df2.groupby(['Ward','Year'])['Arrest'].transform('size')
df2['rate_ward_year']=df2.crime_year_ward/df2.crime_year

df2['mean_crime_year']=df2.groupby('Year')['rate_ward_year'].transform('mean')

df2['unsafe']=(df2.rate_ward_year>df2['mean_crime_year'])*1
df2.drop(columns=['Latitude', 'Longitude', 'crime_year', 'crime_year_ward', 'rate_ward_year', 'mean_crime_year', \

                  'Date','Description','Location Description','Ward'], inplace=True)
df3 = df2.copy()
df2= pd.get_dummies(df2,columns=['Primary Type'])
x=df2.drop(columns='unsafe')

y=df2.unsafe

dt = tree.DecisionTreeClassifier(max_depth = 3)

dt.fit(x,y)
dt_feature_names = list(x.columns)

dt_target_names = np.array(y.unique(),dtype=np.str) 

tree.export_graphviz(dt, out_file='tree.dot', 

    feature_names=dt_feature_names, class_names=dt_target_names,

    filled=True)  

graph = pydotplus.graph_from_dot_file('tree.dot')

Image(graph.create_png())
df2['is_distance_3km']=(df2.distance<3)*1
df2.drop(columns = ['unsafe','distance'], inplace = True)
x=df2.drop(columns='is_distance_3km')

y=df2.is_distance_3km

dt = tree.DecisionTreeClassifier(max_depth = 3)

dt.fit(x,y)
dt_feature_names = list(x.columns)

dt_target_names = np.array(y.unique(),dtype=np.str) 

tree.export_graphviz(dt, out_file='tree.dot', 

    feature_names=dt_feature_names, class_names=dt_target_names,

    filled=True)  

graph = pydotplus.graph_from_dot_file('tree.dot')

Image(graph.create_png())
v=df2.columns.tolist()

cols=[]

for l in v:

    if 'Primary' in l:

        cols.append(l)

df2.groupby(['is_distance_3km'])[cols].mean().T
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(18, 8))

sns.catplot(data=df2,x='Year',y='Primary Type_NARCOTICS',kind ='point',aspect=1,hue='is_distance_3km',ax=ax1)

sns.catplot(data=df2,x='Year',y='Primary Type_THEFT',kind ='point',aspect=1,hue='is_distance_3km',ax=ax2)

plt.close(2)

plt.close(3)
df3['is_distance_3km']=(df3.distance<3)*1
df3 = df3[(df3['Primary Type'] == 'THEFT') | (df3['Primary Type'] == 'NARCOTICS')]
sns.factorplot(data=df3,x='Year',y='is_distance_3km',kind ='bar',aspect=1,hue='Primary Type')