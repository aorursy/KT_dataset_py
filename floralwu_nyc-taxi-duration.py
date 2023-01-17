# Importing the libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import datetime as dt

from datetime import datetime, timedelta,date

from sklearn.model_selection import train_test_split  

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import time
# Importing the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# First of all, take a look at the imported data

print("train as a pd DataFrame")

print(train.info())

print(train.head())

print("test as a pd DataFrame")

print(test.info())

print(test.head())
 #  check  column id, to see if there is any overlapping ids in train and test data set;

train_id = set(train['id'].values)

test_id = set(test['id'].values)

overlap_id = train_id.intersection(test_id)

print("Number of overlaping id in the train and test datasets : ", len(overlap_id))
y = train['trip_duration'] 

print("Longest trip_duration = {}  minutes: " .format( np.max(y.values)//60))

print("Smallest trip_duration = {} minutes: ".format(np.min(y.values)//60))

print("Average trip_duration = {} minutes".format( np.mean(y.values)//60))
#visalize the trip_duration 

f = plt.figure(figsize=(8,6))

#plt.scatter(range(len(y)), np.sort(y.values), alpha=0.5)

plt.scatter(range(len(y)) ,  y , alpha=0.5)

plt.xlabel('Index')

plt.ylabel('Trip duration in minutes')

plt.show()
#exulding outliers

P = np.percentile(y, [0.5, 99.5])

train= train[(train['trip_duration'] > P[0]) & (train['trip_duration']< P[1])]

# redefine y

y = train['trip_duration']
#visalize the improved trip_duration 

f = plt.figure(figsize=(8,6))

#plt.scatter(range(len(y)), np.sort(y.values), alpha=0.5)

plt.scatter(range(len(y)) ,  y , alpha=0.5)

plt.xlabel('Index')

plt.ylabel('Trip duration in minutes')

plt.show()
#visaulize the distribution of y 

f = plt.figure(figsize=(8,6))

plt.hist(y/60)

plt.xlabel('Trip duration in minutes')

plt.ylabel('Frequency')

plt.show()
# data preprocessing on trainning dataset

# takeing care of  datetime type

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])



train['pickup_day'] = train['pickup_datetime'].dt.day

train['pickup_month'] = train['pickup_datetime'].dt.month

train['pickup_weekday'] = train['pickup_datetime'].dt.weekday

train['pickup_hour'] = train['pickup_datetime'].dt.hour



train['drop_day'] = train['dropoff_datetime'].dt.day

train['drop_month'] = train['dropoff_datetime'].dt.month

train['drop_weekday'] = train['dropoff_datetime'].dt.weekday

train['drop_hour'] = train['dropoff_datetime'].dt.hour



# finding out holidays

cal = calendar()

holidays = cal.holidays(start=train['pickup_datetime'].min(), end=train['pickup_datetime'].max())

#df = pd.DataFrame()

train['holiday'] = train['pickup_datetime'].astype('datetime64[ns]').isin(holidays)

 

#construct training dataset as in testing dataset



train1 = train

#train = train1

# First of all, take a look at the imported data

print("train1 as a pd DataFrame")

print(train.info())



train1 = train1.drop('pickup_datetime',axis =1)

train1 = train1.drop('dropoff_datetime',axis =1)

train1 = train1.drop('trip_duration',axis  =1)

train1 = train1.drop('drop_day',axis  =1)

train1 = train1.drop('drop_month',axis  =1)

train1 = train1.drop('drop_weekday',axis  =1)

train1 = train1.drop('drop_hour',axis  =1)

print("train1 after dropping columns")

print(train1.info())



#Encoding categorical data

from sklearn.preprocessing import LabelEncoder

train1 = train1.values

labelencoder_X = LabelEncoder()



train1[:,7] = labelencoder_X.fit_transform(train1[:,7])

train1[:,12] = labelencoder_X.fit_transform(train1[:,12])

#set up X_traian  

X = train1

y = y.values
# Splitting the dataset into the Training set and Test set

# this is for the purpose of evaluating the performance, not for the required predetion 

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_id = X_train[:,0]

X_test_id = X_test[:,0]

X_train = X_train[:,1:]

X_test = X_test[:,1:]

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# Fitting Multiple Linear Regression 

from sklearn.linear_model import LinearRegression# Predicting the Test set results

 

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Predicting the Test set results (this X_test is part of the train data, for performance evaluation popose)

y_pred = regressor.predict(X_test) 

print (y_pred)

print (y_test)
#result = pd.concat(comp, ignore_index=True)

result = pd.DataFrame({'id':X_test_id,'y_test':y_test,'y_pred':y_pred})
# Evaluating the model performance

import statsmodels.formula.api as sm

X = X[:,1:]

X = sc_X.fit_transform(X)

X = np.append(arr = np.ones((1444013, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [ 1, 2, 3, 4, 5,6,7,8,9,10,11,12]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()