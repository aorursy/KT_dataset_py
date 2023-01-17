import os



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as snb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



%matplotlib inline

snb.set({'figure.figsize':(16,8), 'axes.titlesize':30, 'axes.labelsize':20})

#mpl.rcParams('axes.titilesize')=20
TRAINFILEPATH = os.path.join('..', 'input', 'train.csv')
data = pd.read_csv(TRAINFILEPATH, index_col=0)
data.head()
data.shape
data.info()
data.isna().sum()
ax = data['passenger_count'].value_counts(normalize=True).plot.bar();

ax.set_ylabel("Percentage")

ax.set_xlabel("Passenger count")

ax.set_title("Repartition of passenger count");
data[data['passenger_count'] == 0].shape
ax = data['vendor_id'].value_counts(normalize=True).plot.bar()

ax.set_xlabel("Vendor ID")

ax.set_ylabel("Frequency")

ax.set_title("Frequency of vendor ID");
ax = data['store_and_fwd_flag'].value_counts(normalize=True).plot.bar()

ax.set_xlabel("store_and_fwd_flag")

ax.set_ylabel("Frequency")

ax.set_title("Frequency of store_and_fwd_flag");
plt.scatter(data['vendor_id'], data['trip_duration'])

plt.xlabel("Vendor ID");

plt.ylabel("Trip duration")

plt.title("Scatter plot of vendor ID & trip duration");
# changing the date format of pickup_datetime and dropoff_datetime from objectto datetime64

data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
ax = data['pickup_datetime'].dt.year.value_counts(normalize=True, ascending=True,).plot.bar()

ax.set_xlabel("year");

ax.set_ylabel("Frequency")

ax.set_title("Pickup frequency by years ");
ax = data['pickup_datetime'].dt.month.value_counts(normalize=True, ascending=True,).plot.bar()

ax.set_xlabel("month");

ax.set_ylabel("Frequency")

ax.set_title("Pickup frequency by months ");
ax = data["trip_duration"].plot.hist()

ax.set_xlabel("Trip duration")

ax.set_ylabel("Frequency")

ax.set_title("Frequency of trip duration");
ax = data.loc[data['trip_duration'] < 5000, 'trip_duration'].hist(bins=20)

ax.set_xlabel("Trip duration")

ax.set_ylabel("Frequency")

ax.set_title("Frequency of trip duration - a zoom in");
ax = np.log(data["trip_duration"]).hist(bins=50)

ax.set_xlabel("Log of trip duration")

ax.set_ylabel("Frequency")

ax.set_title("Frequency of trip durations log");
snb.boxplot(data["trip_duration"]);
# We keep only the trips with a duration less than = 6 hours (21600 s) in in our dataset

data = data[data["trip_duration"]<21600] 
data.shape
print ("{} trips have been removed. It means {:0.2f}% of the total trips in the dataset.".format(1458644-1456583, (1458644-1456583)*100/1458644))
# create a copy of our dataset in order to have a backup

train = data 
# adding the new columns to the train dataframe

train['month_pickup']=data['pickup_datetime'].dt.month

train['day_pickup']=data['pickup_datetime'].dt.dayofweek

train['hour_pickup']=data['pickup_datetime'].dt.hour

train['minute_pickup']=data['pickup_datetime'].dt.minute

train['second_pickup']=data['pickup_datetime'].dt.second

# deleting the column pickup_datetime from the train dataframe

train = train.drop(columns=['pickup_datetime'])
# This piece of code is inspired from: http://blog.tkbe.org/archive/python-calculating-the-distance-between-two-locations/

# It calculate the "crow flies" distance between two locations 

import math

 

def cosrad(n):

    "Return the cosine of ``n`` degrees in radians."

    return math.cos(math.radians(n))



def distance(row):

    """Calculate the distance between two points on earth.

    """

    lat1 = row['pickup_latitude']

    long1 = row['pickup_longitude']

    lat2 = row['dropoff_latitude']

    long2 = row['dropoff_longitude']

    earth_radius = 6371  # km

    dLat = math.radians(lat2 - lat1)

    dLong = math.radians(long2 - long1)

    a = (math.sin(dLat / 2) ** 2 +

         cosrad(lat1) * cosrad(lat2) * math.sin(dLong / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = earth_radius * c

    return d
# adding the trip distance column to the train dataframe

train['trip_distance']=data.apply(distance, axis=1)
train['trip_duration_log']=data['trip_duration'].apply(np.log)
train.head()
features = ['trip_distance', 'day_pickup', 'hour_pickup', 'minute_pickup', "pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude"]

target = 'trip_duration_log'
X = train[features]

y = train[target]

X.shape, y.shape
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score
# Find in the comment of Enrique Pérez Herrero in: https://www.kaggle.com/marknagelberg/rmsle-function

def rmsle_func(ypred, ytest) :

    assert len(ytest) == len(ypred)

    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))
rmsle = make_scorer(rmsle_func) # Make RMSLE as a scorer
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import LinearRegression
rfr = RandomForestRegressor(n_estimators=15)

#rfr1 =RandomForestRegressor (n_estimators=30, min_samples_leaf=10, min_samples_split=15, max_depth=90, bootstrap=True)

reg = LinearRegression()

sgdr = SGDRegressor()
scores_rfr = cross_val_score(rfr, X, y, cv=5, scoring=rmsle)
print("RMSLE: %0.2f (+/- %0.4f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
scores_reg = cross_val_score(reg, X, y, cv=5, scoring=rmsle)

print("RMSLE: %0.2f (+/- %0.4f)" % (scores_reg.mean(), scores_reg.std() * 2))
scores_sgdr= cross_val_score(sgdr, X, y, cv=5, scoring=rmsle)
print("RMSLE: %0.2f (+/- %0.4f)" % (scores_sgdr.mean(), scores_sgdr.std() * 2))
rfr.fit(X, y)
TESTFILEPATH = os.path.join('..', 'input', 'test.csv')

test = pd.read_csv(TESTFILEPATH)
# Date type transformation

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

# It calculate the "crow flies" distance between two locations 

test['month_pickup']=test['pickup_datetime'].dt.month

test['day_pickup']=test['pickup_datetime'].dt.dayofweek

test['hour_pickup']=test['pickup_datetime'].dt.hour

test['minute_pickup']=test['pickup_datetime'].dt.minute

test['second_pickup']=test['pickup_datetime'].dt.second

test = test.drop(columns=['pickup_datetime'])

# adding the trip distance column

test['trip_distance']=test.apply(distance, axis=1)
test.head()
# test independant variables (features)

test_X = test[features]
predicted_duration_log = rfr.predict(test_X) 
predicted_duration = np.exp(predicted_duration_log) # reverse the log predictions

predicted_duration
my_submission = pd.DataFrame({'id': test['id'], 'trip_duration': predicted_duration})

WORKINGFILEPATH = os.path.join('..', 'working', 'submission.csv')

my_submission.to_csv(WORKINGFILEPATH, index=False)