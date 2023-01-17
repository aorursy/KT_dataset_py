import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from math import sin, cos, sqrt, atan2, radians
import os
import time
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train=train_df.copy()
train_df.head()
#train_df.shape
#train_df.describe()
train_df.isnull().values.any()
# No null values
train_df.info()
#train_df[train_df['passenger_count']>1].count()
#Categorical Features
train_df.describe(include=['O'])
corr = train_df.select_dtypes(include = ['float64', 'int64']).iloc[:, 0:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1, square=True)
train_df.trip_duration.plot.hist()
%matplotlib inline
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['trip_duration'].values+1), axlabel = 'Log(trip_duration)', label = 'log(trip_duration)', bins = 50, color="r")
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()
start=time.time()
train_df['pickup_datetime']=pd.to_datetime(train_df.pickup_datetime)
train_df['dropoff_datetime']=pd.to_datetime(train_df.dropoff_datetime)

train_df.loc[:,'pick_date']=train_df['pickup_datetime'].dt.date
train_df.loc[:,'drop_date']=train_df['dropoff_datetime'].dt.date
print(train_df.head())
end=time.time()
start=time.time()
test_df['pickup_datetime']=pd.to_datetime(train_df.pickup_datetime)
test_df['dropoff_datetime']=pd.to_datetime(train_df.dropoff_datetime)

test_df.loc[:,'pick_date']=test_df['pickup_datetime'].dt.date
test_df.loc[:,'drop_date']=test_df['dropoff_datetime'].dt.date
print(test_df.head())
end=time.time()
start=time.time()
train_sam=train_df.copy()
train_df.loc[:,'pick_time']=train_df['pickup_datetime'].dt.time
train_df.loc[:,'drop_time']=train_df['dropoff_datetime'].dt.time
print(train_df.head())
end=time.time()
start=time.time()
test_sam=train_df.copy()
test_df.loc[:,'pick_time']=test_df['pickup_datetime'].dt.time
test_df.loc[:,'drop_time']=test_df['dropoff_datetime'].dt.time
print(test_df.head())
end=time.time()
start=time.time()
train_sam=train_df.copy()
train_df.loc[:,'pick_day']=train_df['pickup_datetime'].dt.day
train_df.loc[:,'drop_day']=train_df['dropoff_datetime'].dt.day
train_df.loc[:,'pick_month']=train_df['pickup_datetime'].dt.month
train_df.loc[:,'drop_month']=train_df['dropoff_datetime'].dt.month

end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
start=time.time()
test_sam=test_df.copy()
test_df.loc[:,'pick_day']=test_df['pickup_datetime'].dt.day
test_df.loc[:,'drop_day']=test_df['dropoff_datetime'].dt.day
test_df.loc[:,'pick_month']=test_df['pickup_datetime'].dt.month
test_df.loc[:,'drop_month']=test_df['dropoff_datetime'].dt.month

end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
start=time.time()
train_sam=train_df.copy()
train_df.loc[:,'pick_day_week']=train_df['pickup_datetime'].dt.weekday
train_df.loc[:,'drop_day_week']=train_df['dropoff_datetime'].dt.weekday
train_df.loc[:,'pick_year']=train_df['pickup_datetime'].dt.year
train_df.loc[:,'drop_year']=train_df['dropoff_datetime'].dt.year

print(train_df.head())
end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
start=time.time()
test_sam=train_df.copy()
test_df.loc[:,'pick_day_week']=test_df['pickup_datetime'].dt.weekday
test_df.loc[:,'drop_day_week']=test_df['dropoff_datetime'].dt.weekday
test_df.loc[:,'pick_year']=test_df['pickup_datetime'].dt.year
test_df.loc[:,'drop_year']=test_df['dropoff_datetime'].dt.year

print(test_df.head())
end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
start=time.time()
train_sam=train_df.copy()
train_df.loc[:,'pick_hour']=train_df['pickup_datetime'].dt.hour
train_df.loc[:,'drop_hour']=train_df['dropoff_datetime'].dt.hour


print(train_df.head())
end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
start=time.time()
test_sam=train_df.copy()
test_df.loc[:,'pick_hour']=test_df['pickup_datetime'].dt.hour
test_df.loc[:,'drop_hour']=test_df['dropoff_datetime'].dt.hour


print(test_df.head())
end=time.time()
print("Time Taken by above cell is {}.".format(end - start))
train_df.info()
train_df=train_df.drop(['pickup_datetime','dropoff_datetime'],axis=1)
train_df=train_df.drop(['pick_year','drop_year'],axis=1)
sns.set(style="white")

# Generate a large random dataset
train_sam = train_df.copy()

# Compute the correlation matrix
corr = train_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 13))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# converting trip duration in min.
train_df['trip_duration']=train_df['trip_duration'].astype('float64')
f=lambda x: x/60

train_df['trip_duration']=train_df['trip_duration'].apply(f)
print(train_df['trip_duration'].head(5))
print(train_df['trip_duration'].max())
train_df.info()
# passenger count vs trip duration
train_df.plot.scatter(x='passenger_count',y='trip_duration')
#train_df[train_df['trip_duration']<3000].plot.hexbin(x='trip_duration',y='passenger_count',gridsize=150)
train_df[train_df['trip_duration']>30000]=0
train_df[train_df['trip_duration']>30000]
#removing entries for which trip_duration>30000 min. i.e. outliers

#removing entries for which passenger_count>6 i.e. outliers
train_df[train_df['passenger_count']>6]=0
train_df[train_df['passenger_count']>6]
#removing rows where passenger count is 0
#train_df[train_df['passenger_count']<1]=0
sl=train_df[train_df['passenger_count']<1].index
#train_df[train_df['passenger_count']<1].iloc[:,:1]

train_df=train_df.drop(sl)
train_df[train_df['passenger_count']<1]
#train_df[(train_df['passenger_count']<1).]

train_df.plot.scatter(x='passenger_count',y='trip_duration')
print(train_df['passenger_count'].value_counts())
(train_df['passenger_count'].value_counts().sort_index()).plot.bar()
train_df['passenger_count'].value_counts().sort_index().plot.line()
#plot showing percentages 
(train_df['passenger_count'].value_counts().sort_index()/len(train_df)).plot.bar()
(train_df[train_df['trip_duration']<30]['trip_duration']).plot.hist()
#train_df['trip_duration'].value_counts().sort_index()
#train_df[train_df['trip_duration']<0.5]
#train_df['pickup_hour']=0
#for i in range(len(train_df)):
#    train_df['pickup_hour'][i]=train_df['pickup_datetime'][i][12:]
#train_df.head() 
#train_df[['vendor_id','trip_duration']].groupby(['vendor_id'],as_index=False).mean()
train_df[['store_and_fwd_flag','trip_duration']].groupby(['store_and_fwd_flag'],as_index=False).mean()
    
#droping vendor_id and store_and_fwd_flag columns
train_df=train_df.drop(['vendor_id','store_and_fwd_flag'],axis=1)
train_df.info()
train_df[['passenger_count','trip_duration']].groupby(['passenger_count'],as_index=False).mean()
train_df['distance']=0.0
train_df.info()
len(train_df)
#t1=train_df.iloc[0:100000,]
#len(t1)
#t1.head()
train_df=train_df.reset_index()
train_df.tail()
#train_df.drop['index','level_0']
#train_df.tail()





train_df=train_df.drop(columns=['index'])
train_df.tail()
t1=train_df.iloc[0:200000,]
t2=train_df.iloc[200000:400000,]
t3=train_df.iloc[400000:600000,]
t4=train_df.iloc[600000:800000,]
t5=train_df.iloc[800000:1000000,]
t6=train_df.iloc[1000000:1200000,]
t7=train_df.iloc[1200000:,]


#

R = 6373.0
#t1=train_df.loc[:,('pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')]
'''
for i in range(len(train_df)):
    pa,po,da,do=train_df.loc[i,['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]
   # po=train_df['pickup_longitude'][i]
   # pa=train_df['pickup_latitude'][i]
    #do=train_df['dropoff_longitude'][i]
    #da=train_df['dropoff_latitude'][i]
    
    
    lat1 = radians(pa)
    lon1 = radians(po)
    lat2 = radians(da)
    lon2 = radians(do)
    
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    train_df.loc[i,['distance']]=distance
    '''
    
def dist(srs):
    #srs.points = srs.points - review_points_mean
    lat1 = radians(srs.pickup_latitude)
    lon1 = radians(srs.pickup_longitude)
    lat2 = radians(srs.dropoff_latitude)
    lon2 = radians(srs.dropoff_longitude)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    srs.distance=distance
    return srs
t1=t1.apply(dist, axis='columns')
t1.head()


t2=t2.apply(dist, axis='columns')
t2.head()

t3=t3.apply(dist, axis='columns')
t3.head()
t4=t4.apply(dist, axis='columns')
t4.head()
t5=t5.apply(dist, axis='columns')
t5.head()
t6=t6.apply(dist, axis='columns')
t6.head()
t7=t7.apply(dist, axis='columns')
t7.head()
#t10=pd.merge(t1,t2,how='outer')
train_df=pd.concat([t1,t2,t3,t4,t5,t6,t7])
print(len(t1))
print(len(t2))
print(len(t3))
print(len(train_df))
train_df.tail()
#plotting distance against duration
sns.jointplot(x='distance', y='trip_duration', data=train_df[train_df['distance']<200])
corr = train_df.select_dtypes(include = ['float64', 'int64']).iloc[:, 0:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1, square=True)
train_df.drop(['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'],axis=1,inplace=True)
train_df.head()
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
y=train_df.trip_duration
predictors=['passenger_count','distance','pick_day','drop_day','pick_hour','drop_hour','pick_day_week','drop_day_week','pick_month','drop_month']
x=train_df[predictors]
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)

from sklearn.ensemble import RandomForestRegressor
forest_model=RandomForestRegressor()
forest_model.fit(train_x,train_y)
trip_pred=forest_model.predict(val_x)
print(mean_absolute_error(val_y,trip_pred))


test_df['distance']=0.0
test_df=test_df.apply(dist, axis='columns')
test_df.head()
train_y=train_df.trip_duration
train_x=train_df[predictors]
test_x=test_df[predictors]
#test_df['trip_duration']=test_df['trip_duration'].apply(f)
forest_model.fit(train_x,train_y)
trip_pred=forest_model.predict(test_x)



test_df['trip_duration']=trip_pred
test_df.trip_duration.head()
f=lambda x: x*60

test_df['trip_duration']=train_df['trip_duration'].apply(f)

test_df.trip_duration.head()
my_submission = pd.DataFrame({'id': test_df.id, 'trip_duration': test_df.trip_duration})
my_submission.to_csv('submission.csv', index=False)
