# imports

import math

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns

sns.set()

import sklearn

from __future__ import print_function

from IPython.display import Image

from IPython.display import display

from IPython.display import HTML

from sklearn import metrics

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn import preprocessing
# Load data from disk and preview data

dt1 = pd.read_csv("../input/911.csv")

dt1.head(3)
# restrict data to 2016 to get full year picture (data contain also events from Dec 2015 and Jan 2017)

# remove unnecessary columns

# we are interested only in accident time, title and geolocation data

# preview limited data

dt1 = dt1[dt1['timeStamp'].str.contains('2016', na = False)]

dt2 = dt1.drop(["desc","addr","e"],axis=1)

print ("Dataset shape :",dt1.shape)

dt2.head(3)
# check if we have rows with empty data in the dataset

print ("lat empty count :", dt2['lat'].isnull().sum())

print ("lng empty count :", dt2['lng'].isnull().sum())

print ("zip empty count :", dt2['zip'].isnull().sum())

print ("title empty count :", dt2['title'].isnull().sum())

print ("timeStamp empty count :", dt2['timeStamp'].isnull().sum())

print ("twp empty count :", dt2['twp'].isnull().sum())
# cleaning columns with empty values, we can achieve our goals without them 

# it is better to remove columns than get rid of around 18k of events

dt3 = dt2.drop(["zip","twp"],axis=1)

dt3.head(3)
# let's check the statistical properties of the numerical data

dt3.describe()
# it seems we have some outliers in the geographical data. 

# Montgomery County (PA) does not have any locations with min lat 30.333596 or min lng -95.595595. 

# there must be some human (data entry operator) mistakes and we need to identify them

outliers = dt3.loc[((dt3['lat'] < 39.00) | (dt3['lat'] > 41.00)) & 

                   ((dt3['lng'] < -77.00) | (dt3['lng'] > -74.00))]

print ("Outliers :\n",outliers)
# we have just a few outliers, removing them will not affect the result. Lets filter out the outliers

dt4 = dt3.loc[((dt3['lat'] > 39.00) & (dt3['lat'] < 41.00)) & 

              ((dt3['lng'] > -77.00) & (dt3['lng'] < -74.00))]

# and describe the dataset again to check effect of the cleaning

dt4.describe()
# we want to get event type from its title

# first we list unique values in title

titles_unique = pd.DataFrame(dt4.title.unique())

titles_unique = titles_unique.sort_values([0],ascending =  True)

print ("Unique titles size :",len(titles_unique))

titles_unique.head(5)
# as we have more than 100 unique categories of events, let's  modify dataset and assign

# to each event only the master category (the one before the colon)

dt5 = dt4.copy()

dt5['category'],dt5['category2'] = dt5['title'].str.split(':',1).str

dt5 = dt5.drop(['title','category2'],axis = 1)

cat_unique = pd.DataFrame(dt5.category.unique())

cat_unique = cat_unique.sort_values([0],ascending =  True)

cat_unique.head()
# We now have the dictionary of the unique categories. We can change strings in dataset to the category numbers. 

# That is necessary as we want to feed the Machine Learning model with this dataset

# and it must contain only numeric values. Here is the mapping:

# 0 = EMS (Emergency Medical Services)

# 1 = FIRE

# 2 =  TRAFFIC

CATEGORIES = {'EMS':0,'Fire':1,'Traffic':2}

dt5['category'].replace(CATEGORIES,inplace=True)

dt5.head(3)
# now we want to parse timestamp to get more information from it.

# we will extend the dataset with more time related values

# hours_range allows us to split day into several periods, each hours_range long

hours_range = 8

dt6 = dt5

dt6['datetime'] = pd.to_datetime(dt5['timeStamp'])

dt6['year'] = dt5['datetime'].dt.year

dt6['month'] = dt5['datetime'].dt.month

dt6['day'] = dt5['datetime'].dt.day

dt6['day_part'] = np.floor(dt5['datetime'].dt.hour/hours_range)

dt6['day_part'] = dt5.day_part.astype(int)

dt6['dayofweek'] = dt5['datetime'].dt.dayofweek

dt6['week'] = dt5['datetime'].dt.week

#let's describe the dat again

dt6.describe()
# the geo coordinates have limited range

# we want to split the whole location into the geo grid

# epsilon is to extend the upper bound minimally 

# to avoid assigning locations at the end of the range to new slot beyond the grid

epsilon = 0.0001

lat_max = dt6['lat'].max() + epsilon

lat_min = dt6['lat'].min()

lat_range = lat_max - lat_min

print ("Latitude min-max: <",lat_min,lat_max,"> | range :",lat_range)

lng_max = dt6['lng'].max() + epsilon

lng_min = dt6['lng'].min()

lng_range = lng_max - lng_min

print ("Longitude min-max: <",lng_min,lng_max,"> | range :",lng_range)
# Let's then split the area set by these coordinates into an grid

# we will divide the lat and lng range, thus creating grid of rectangles

lat_split = 5 # number of horizontal parts

lng_split = 7 # number of vertical parts

lat_hop = lat_range/lat_split # lat divided to N parts gives us length of one part

print ("Lat hop : ",lat_hop)

lng_hop = lng_range/lng_split # lng divided to N parts gives us length of one part

print ("Lng hop : ",lng_hop)

# now we need to assign coordinates to proper geogrid squares

dt6['lat_grid'] = (np.floor(((dt6['lat']-lat_min)/lat_hop)))

dt6['lng_grid'] = (np.floor(((dt6['lng']-lng_min)/lng_hop)))

dt6.lat_grid = dt6.lat_grid.astype(int)

dt6.lng_grid = dt6.lng_grid.astype(int)

dt7 = dt6.drop(['lat','lng'],axis = 1)

dt7 = dt6

dt7.head(3)
# let's check number of events per month

fig, ax = plt.subplots(figsize=(7,3))  

ax = sns.countplot(x="month", data=dt7,ax=ax)
# let's check number of events per day of the week

fig, ax = plt.subplots(figsize=(5,3))

ax = sns.countplot(x="dayofweek", data=dt7)

ax.axes.set_xticklabels(["MON", "TUE","WED","THU","FRI","SAT","SUN"])

pass
#let's see the size of each category (class)

# 0 = EMS (Emergency Medical Services), 1 = FIRE, 2 =  TRAFFIC

fig, ax = plt.subplots(figsize=(5,3))

ax = sns.countplot(x="category", data=dt7)

ax.axes.set_xticklabels(["EMS","FIRE","TRAFFIC"])

pass
# lets check the time impact on the events

dt_timegrid = dt7.groupby(['dayofweek','day_part']).size().reset_index(name='count')

dt_timeheatmap = dt_timegrid.pivot(index='day_part', columns='dayofweek', values='count')

# generate heatmap

fig, ax = plt.subplots(figsize=(5,3))

ax = sns.heatmap(dt_timeheatmap,annot=True, fmt="d",cbar=False)

ax.invert_yaxis()

ax.axes.set_yticklabels(["16-24 h","08-16 h","00-08 h"])

ax.axes.set_xticklabels(["MON", "TUE","WED","THU","FRI","SAT","SUN"])

pass
# now we can visualize our data on the geogrid.

dt_geogrid = dt7.groupby(['lat_grid','lng_grid']).size().reset_index(name='count')

dt_geoheatmap = dt_geogrid.pivot(index='lat_grid',columns='lng_grid', values='count')

# generate heatmap

fig, ax = plt.subplots(figsize=(5,5))  

ax = sns.heatmap(dt_geoheatmap,annot=True,fmt=".0f",cbar=False)

ax.invert_yaxis()

sns.plt.show()

print ("Longitude min-max: <",lng_min,lng_max,"> | range :",lng_range)

print ("Latitude min-max: <",lat_min,lat_max,"> | range :",lat_range)

#draw reference map

print ("\nUS PA Montgomery County Reference map") 

print ("Map source: OpenStreetMap.org, Map license: Open Data Commons Open Database License (ODbL).")

# reference grid image is on my blog: 

# http://machinelearningexp.com/machine-learning-regression-911-calls/

# Kaggle does not allow to upload additional files yet

print ("See http://machinelearningexp.com/machine-learning-regression-911-calls/")
# reorganize table to have mor intuitive order of the features

final_columns = ["month","week","dayofweek","day","day_part","lat_grid","lng_grid","category"]

dt7 = dt6[final_columns]

dt7.head(3)
# let's describe the data again

dt7.describe()
# create separate datasets for categories and group them by all parameters to get count of events for a given group

groupby_list = ['month','week','dayofweek','day','day_part','lat_grid','lng_grid']

dt_cat = dict() # holder for subdatasets with categories. 

for item in CATEGORIES:

    dt_temp = dt7.loc[(dt7['category'] == CATEGORIES[item])]

    dt_cat[item] =  dt_temp.groupby(groupby_list).size().reset_index(name='count')

dt_cat['ALL'] = dt7.groupby(groupby_list).size().reset_index(name='count') # All data, without category grouping

dt_cat['ALL'].head(3) 
dt_cat['ALL'].describe()
# let's now create a function that will split data into train and test sets and run regresion algorithm on the data



def run_regression(name,input_dt):

    X = input_dt.iloc[:,[0,1,2,3,4,5,6]]

    Y = input_dt.iloc[:,[7]]

    Y = Y.values.reshape(len(X))

    validation_size = 0.20

    seed = 7

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = validation_size,random_state = seed)

    model = GradientBoostingRegressor(n_estimators=200, 

                                      learning_rate=0.1, max_depth=5, random_state=0, loss='ls', warm_start =  True)

    model.fit(X_train,Y_train)

    return name,model,r2_score(Y_test, model.predict(X_test))



# run model for all categories and put results into the table.

# also save trained models for later use

results_table = [["CATEGORY","R2 SCORE"]]

trained_models = dict() # holder for trained models

for item in dt_cat:

    results = run_regression(item,dt_cat[item])

    results_table.append([item,results[2]])

    trained_models[item] =  results[1]



for row in results_table:

    print (row)
# we will use trained GradientBoostingRegressor model to estimate 911 calls

# in a single day of 2017, based on the 2016 year data

# we need to generate list containing all time slots in a single day and "active" geogrid locations

# the selected date will be 19 May 2017 (arbitrary date)

# note we cannot use all geogrid locations as not for all we have the data

# and model will not be able to predict anything meaningful for them

# the county map does not cover the whole grid. 

# So we will use the previous dt_geogrid variable to get "active" locations

singleday_dt = []

# record structure is month,week,dayofweek,day,day_part,lat_grid,lng_grid

row_base = [5,20,4,19] #base row with date 19 May 2017, Wednesday. Change it to get another day.

for day_idx in range(int(24/hours_range)):

    for idx,row in dt_geogrid.iterrows():

        singleday_dt.append(row_base+[day_idx,row['lat_grid'],row['lng_grid']]) 

singleday_dt = pd.DataFrame(singleday_dt,columns=final_columns[:7])

singleday_dt.head(3)
# we will pass generated data to scikit-learn model predict method to see the result

predictions_all = trained_models['ALL'].predict(singleday_dt)

singleday_dt_full = singleday_dt

singleday_dt_full['events'] = predictions_all

print ("Total number of 911 events in selected day is : ", round(singleday_dt_full['events'].sum()))
# now we can visualize our data for 19 May 2017 on the map.

dt_geogrid = singleday_dt_full.groupby(['lat_grid','lng_grid']).agg({'events': np.sum}).reset_index()

dt_geoheatmap = dt_geogrid.pivot(index='lat_grid', columns='lng_grid', values='events')

# generate heatmap

fig, ax = plt.subplots(figsize=(5,5))  

ax = sns.heatmap(dt_geoheatmap,annot=True,fmt=".0f",cbar=False)

ax.invert_yaxis()

sns.plt.show()

fig = ax.get_figure()

print ("US PA Montgomery County Reference map") 

print ("Map source: OpenStreetMap.org, Map license: Open Data Commons Open Database License (ODbL).")

# reference grid image is on my blog: 

# http://machinelearningexp.com/machine-learning-regression-911-calls/

# Kaggle does not allow to upload additional files yet

print ("See http://machinelearningexp.com/machine-learning-regression-911-calls/")
data_timeevents = singleday_dt_full.groupby(['day_part']).agg({'events': np.sum}).reset_index()

fig, ax = plt.subplots(figsize=(5,3))

ax = sns.barplot(x="day_part", y="events", data=data_timeevents)

ax.axes.set_xticklabels(["00-08 h","08-16 h","16-24 h"])

pass