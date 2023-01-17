import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for visualization

import seaborn as sns # for statistical visualization

plt.style.use('ggplot') # Set style for plotting
# Read 10,000,000 rows so that the kernel won't died easily

train = pd.read_csv('../input/train.csv', nrows = 5000000)
# Look at the top 3 rows of data

train.head(3)
# Structure and data types

train.info()
# Statistical analysis overlook

pd.set_option('float_format', '{:f}'.format) # Print entire number instead of x + ye



train.describe()
train = train[train.fare_amount > 0]
train.shape
test = pd.read_csv('../input/test.csv')



test['longitude_diff'] = test['dropoff_longitude'] - test['pickup_longitude']

test['latitude_diff'] = test['dropoff_latitude'] - test['pickup_latitude']
test.describe()
# Mean + 3 * std

train = train.loc[train.fare_amount < 35]
train.shape
# So I set up a longitude range for the ride

train = train.loc[train.pickup_longitude < -72.5]

train = train.loc[train.pickup_longitude > -74.5]
train.shape
# And a latitude range for the ride

train = train.loc[train.pickup_latitude < 42]

train = train.loc[train.pickup_latitude > 40]
train = train.loc[train.dropoff_longitude < -72.5]

train = train.loc[train.dropoff_longitude > -74.5]
train = train.loc[train.dropoff_latitude < 42]

train = train.loc[train.dropoff_latitude > 40]
train.shape
train['longitude_diff'] = train['dropoff_longitude'] - train['pickup_longitude']



train['latitude_diff'] = train['dropoff_latitude'] - train['pickup_latitude']
train = train.loc[train.longitude_diff > -0.9]

train = train.loc[train.longitude_diff < 0.5]



train = train.loc[train.latitude_diff > -0.7]

train = train.loc[train.latitude_diff < 0.3]
train.shape
train = train.loc[train.passenger_count > 0]

train = train.loc[train.passenger_count <= 6]
train.shape
train.head()
train['year'] = train.pickup_datetime.apply(lambda x: x[:4])

test['year'] = test.pickup_datetime.apply(lambda x: x[:4])
train['month'] = train.pickup_datetime.apply(lambda x: x[5:7])

test['month'] = test.pickup_datetime.apply(lambda x: x[5:7])
train['hour'] = train.pickup_datetime.apply(lambda x: x[11:13])

test['hour'] = test.pickup_datetime.apply(lambda x: x[11:13])
import datetime



train['pickup_datetime'] = train.pickup_datetime.apply(

    lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))

test['pickup_datetime'] = test.pickup_datetime.apply(

    lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
train['day_of_week'] = train.pickup_datetime.apply(lambda x: x.weekday())

test['day_of_week'] = test.pickup_datetime.apply(lambda x: x.weekday())
train['pickup_date'] = train.pickup_datetime.apply(lambda x: x.date())

test['pickup_date'] = test.pickup_datetime.apply(lambda x: x.date())
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()

holidays = cal.holidays(start='2009-01-01', end='2015-12-31').to_pydatetime()



train['holidat_or_not'] = train.pickup_datetime.apply(lambda x: 1 if x in holidays else 0)

test['holidat_or_not'] = test.pickup_datetime.apply(lambda x: 1 if x in holidays else 0)
train = train.drop(['key','pickup_datetime','pickup_date'],axis=1)

test = test.drop(['key','pickup_datetime','pickup_date'],axis=1)
train.info()
train['year'] = train['year'].astype('int')

train['hour'] = train['hour'].astype('int')



test['year'] = test['year'].astype('int')

test['hour'] = test['hour'].astype('int')
train.head()
plt.scatter(train['pickup_longitude'],train['pickup_latitude'],alpha=0.2)
plt.scatter(train['dropoff_longitude'],train['dropoff_latitude'],alpha=0.2)
from sklearn.preprocessing import StandardScaler
sample = train[:50000][['pickup_longitude','pickup_latitude']]
scaler = StandardScaler()

sample = scaler.fit_transform(sample)
from scipy.cluster.vq import kmeans



distortions = []

num_clusters = range(2, 25)



# Create a list of distortions from the kmeans function

for i in num_clusters:

    cluster_centers, distortion = kmeans(sample,i)

    distortions.append(distortion)



# Create a data frame with two lists - num_clusters, distortions

elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})



# Creat a line plot of num_clusters and distortions

sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)

plt.xticks(num_clusters)

plt.show()
sample = train[:50000][['dropoff_longitude','dropoff_latitude']]
distortions = []

num_clusters = range(2, 25)



# Create a list of distortions from the kmeans function

for i in num_clusters:

    cluster_centers, distortion = kmeans(sample,i)

    distortions.append(distortion)



# Create a data frame with two lists - num_clusters, distortions

elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})



# Creat a line plot of num_clusters and distortions

sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)

plt.xticks(num_clusters)

plt.show()
from sklearn.cluster import KMeans



model = KMeans(n_clusters = 5)

model.fit(train[['pickup_longitude','pickup_latitude']])

pickup_labels = model.predict(train[['pickup_longitude','pickup_latitude']])
train['pickup_cluster'] = pickup_labels



pickup_cluster = pd.get_dummies(train['pickup_cluster'],prefix='pickup_cluster',drop_first=True)



train = pd.concat([train,pickup_cluster],axis=1).drop('pickup_cluster',axis=1)
model2 = KMeans(n_clusters = 8)

model2.fit(train[['dropoff_longitude','dropoff_latitude']])

dropoff_labels = model2.predict(train[['dropoff_longitude','dropoff_latitude']])
train['dropoff_cluster'] = dropoff_labels



dropoff_cluster = pd.get_dummies(train['dropoff_cluster'],prefix='dropoff_cluster',drop_first=True)



train = pd.concat([train,dropoff_cluster],axis=1).drop('dropoff_cluster',axis=1)
train['pickup_cluster'] = pickup_labels

train['dropoff_cluster'] = dropoff_labels
train.groupby(['pickup_cluster','dropoff_cluster'])['fare_amount'].agg(['mean','std', 'count'])
pickup_test_labels = model.predict(test[['pickup_longitude','pickup_latitude']])



test['pickup_cluster'] = pickup_test_labels

pickup_cluster = pd.get_dummies(test['pickup_cluster'],prefix='pickup_cluster',drop_first=True)

test = pd.concat([test,pickup_cluster],axis=1).drop('pickup_cluster',axis=1)
dropoff_test_labels = model2.predict(test[['dropoff_longitude','dropoff_latitude']])



test['dropoff_cluster'] = dropoff_test_labels

dropoff_cluster = pd.get_dummies(test['dropoff_cluster'],prefix='dropoff_cluster',drop_first=True)

test = pd.concat([test,dropoff_cluster],axis=1).drop('dropoff_cluster',axis=1)
train.head()
train.shape
train = pd.get_dummies(train,columns=['month','hour','day_of_week'],drop_first=True)

test = pd.get_dummies(test,columns=['month','hour','day_of_week'],drop_first=True)
train.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
train.head()
train = train.drop(['pickup_cluster','dropoff_cluster'],axis=1)
X = train.drop('fare_amount',axis=1)

y = train[['fare_amount']]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestRegressor(n_estimators=30,min_samples_split=5,min_samples_leaf=3,random_state=21)



clf.fit(X_train,y_train)
predictions = clf.predict(X_test)



error = np.sqrt(metrics.mean_squared_error(y_test,predictions))

print(error)
features = X.columns[:X.shape[1]]

importances = clf.feature_importances_

indices = np.argsort(importances)



plt.figure(figsize=(6, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
predictions = clf.predict(test)



submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = predictions

submission.to_csv('submission.csv',index=False)