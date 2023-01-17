# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', nrows = 1000000)
# out of the 55million + data, i have picked a sample and will be working with it

test = pd.read_csv('../input/test.csv')

train.head()
test.head()
train.shape
test.shape
#We check the datatypes 
train.dtypes.value_counts()
test.dtypes.value_counts()
train.isnull().sum()
#I will simply drop the nan columns
train = train.dropna()
train.isnull().sum()
#The we check for zeros in our train and test data
(train ==0).astype(int).sum()
train = train.loc[~(train==0).any(axis =1)]
#we take a look at what we have done
#(train ==0).astype(int).sum()
(train ==0).astype(int).sum()
#we started out with 1 million data point. Lets see what we have now.
train.shape
train.describe()


train.describe()
train.dtypes.value_counts()
#lets take care of the object data first
object_data = train.dtypes == np.object
categoricals = train.columns[object_data]
categoricals

#I will drop the key column since i do not really need it 
train.drop('key', axis = 1, inplace = True)
train.head()
import datetime as dt

def date_extraction(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.day
    data['hour'] = data['pickup_datetime'].dt.hour
    data = data.drop('pickup_datetime', axis = 1, inplace = True)
    
    return data
    
#Apply this to both the train and the test data
date_extraction(train)
#test = date_extraction(test)
    
train.head()
date_extraction(test)
test.head()


#First i will define the Haversine function
#radii of earth in meters = 6371e3 meters
# def long_lat_distance (x):
#     x['Longitude_distance'] = x['pickup_longitude'] - x['dropoff_longitude']
#     x['Latitude_distance'] = x['pickup_latitude'] - x['dropoff_latitude'] 
    
#     return x   
def long_lat_distance (x):
    x['Longitude_distance'] = np.radians(x['pickup_longitude'] - x['dropoff_longitude'])
    x['Latitude_distance'] = np.radians(x['pickup_latitude'] - x['dropoff_latitude']) 
    x['distance_travelled/10e3'] = ((x['Longitude_distance']**2 + x['Latitude_distance']**2)**0.5) *1000
    return x   
for x in [train, test]:
    long_lat_distance(x)
    
train.head()

def harvesine(x):
    #radii of earth in meters = 
    r = 6371000 
    d = x['distance_travelled/10e3']
    theta_1 = np.radians(x['dropoff_latitude'])
    theta_2 = np.radians(x['pickup_latitude'])
    lambda_1 = np.radians(x['dropoff_longitude'])
    lambda_2 = np.radians(x['dropoff_longitude'])
    theta_diff = x['Longitude_distance']
    lambda_diff = x['Latitude_distance']
    
    a = np.sin(theta_diff/2)**2 + np.cos(theta_1)*np.cos(theta_2)*np.sin(lambda_diff/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    x['harvesine/km'] = (r * c)/1000

for x in [train, test]:
    harvesine(x)
    
train.head()
train.dtypes.value_counts()
# #not sure if this is rrally necessary. will have to see

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(train)

train.head()
test.head()
train.describe()
print('Are there any nulls\nan in the train data: ')
print(train.isnull().sum())

print('\nAre there any nulls\nans in the test data: ')
print(test.isnull().sum())

#as we can see there are 3 nulls in the train data. Lets replace them with the mean
train['harvesine/km'] = train['harvesine/km'].fillna(train['harvesine/km'].median())
from sklearn.ensemble import RandomForestRegressor

#split the train features 
feature_cols = [x for x in train.columns if x!= 'fare_amount']
X = train[feature_cols]
y = train['fare_amount']

correlations = X.corrwith(y)
correlations = abs(correlations*100)
correlations.sort_values(ascending = False, inplace= True)

correlations
#lets plot and see what we've got
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation');
train.head()

#From the diagram above, i will use the 5 most important features

train_1 = train.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

train_1.head()
train_1['harvesine/km'] =train_1['harvesine/km'].round(2) 
train_1['distance_travelled/10e3'] =train_1['distance_travelled/10e3'].round(2) 

train_1.head()

# #not sure if this is rrally necessary. will have to see

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(train_1)
train_1.describe()

test_1 = test.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

test_1.head()
from sklearn.model_selection import train_test_split
feat_cols = [x for x in train_1.columns if x!= 'fare_amount']
X_1 = train_1[feat_cols]
y_1 = train_1['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.25, random_state = 42)
#Random forest
rf = RandomForestRegressor(n_estimators = 100, max_features = 5)
rf = rf.fit(X_train, y_train)

test.head()
final_prediction = rf.predict(X_test)
test_1.drop('key', axis = 1, inplace = True)
test_1.head()
#random forest
final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame({'key': test.key, 'fare_amount': final_prediction})
NYCtaxiFare_submission.to_csv('NYCtaxiFare_prediction.csv', index = False)
NYCtaxiFare_submission.head()

