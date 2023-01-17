import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
manhattan_taxi = pd.read_csv('../input/taxi-data-set/manhattan_taxi.csv')

manhattan_taxi.head(5)
def pickup_scatter(t):

    plt.scatter(t['pickup_lon'], t['pickup_lat'], s=2, alpha=0.2)

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')

    plt.title('Pickup locations')



plt.figure(figsize=(8, 16))

pickup_scatter(manhattan_taxi);
manhattan_taxi['date'] = pd.to_datetime(manhattan_taxi['pickup_datetime']).dt.date

manhattan_taxi.head()
date_ride_count = manhattan_taxi.groupby('date').count()

date_ride_count = pd.DataFrame(date_ride_count['pickup_datetime']).reset_index().rename(columns={'pickup_datetime': 'ride count'})

plt.plot_date(date_ride_count['date'], date_ride_count['ride count'])

plt.xticks(rotation=70)

plt.xlabel('date')

plt.ylabel('ride count')

plt.title('Ride Count by Date');
from datetime import date



atypical = [1, 2, 3, 18, 23, 24, 25, 26]

typical_dates = [date(2016, 1, n) for n in range(1, 32) if n not in atypical]

typical_dates



final_taxi = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]
import sklearn.model_selection



train, test = sklearn.model_selection.train_test_split(final_taxi, train_size=0.8, test_size=0.2, random_state=42)

print('Train:', train.shape, 'Test:', test.shape)
train_sorted_dates = train.sort_values('date')

plt.figure(figsize=(8,4))

sns.boxplot(data=train_sorted_dates, x='date', y='duration')

plt.xticks(rotation=90)

plt.title('Duration by date');
def speed(t):

    """Return a column of speeds in miles per hour."""

    return t['distance'] / t['duration'] * 60 * 60



def augment(t):

    """Augment a dataframe t with additional columns."""

    u = t.copy()

    pickup_time = pd.to_datetime(t['pickup_datetime'])

    u.loc[:, 'hour'] = pickup_time.dt.hour

    u.loc[:, 'day'] = pickup_time.dt.weekday

    u.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)

    u.loc[:, 'period'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])

    u.loc[:, 'speed'] = speed(t)

    return u

    

train = augment(train)

test = augment(test)

train.iloc[0,:]
period_1 = train[train['period'] == 1]['speed']

period_2 = train[train['period'] == 2]['speed']

period_3 = train[train['period'] == 3]['speed']



plt.figure(figsize=(8,6))

sns.distplot(period_1, label='Early Morning')

sns.distplot(period_2, label='Daytime')

sns.distplot(period_3, label='Night')

plt.title('Distribution of Speed per Period');

plt.legend();
D = train[['pickup_lon', 'pickup_lat']].to_numpy()

pca_n = len(train)

pca_means = np.mean(D, axis=0)

X = (D - pca_means) / np.sqrt(pca_n)

u, s, vt = np.linalg.svd(X, full_matrices=False)



def add_region(t):

    """Add a region column to t based on vt above."""

    D = t[['pickup_lon', 'pickup_lat']].to_numpy()

    assert D.shape[0] == t.shape[0], 'You set D using the incorrect table'

    X = (D - pca_means) / np.sqrt(pca_n) 

    first_pc = X @ vt.T[0]

    t.loc[:,'region'] = pd.qcut(first_pc, 3, labels=[0, 1, 2])

    

add_region(train)

add_region(test)
plt.figure(figsize=(8, 16))

for i in [0, 1, 2]:

    pickup_scatter(train[train['region'] == i])
from sklearn.preprocessing import StandardScaler



num_vars = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance']

cat_vars = ['hour', 'day', 'region']



scaler = StandardScaler()

scaler.fit(train[num_vars])



def design_matrix(t):

    """Create a design matrix from taxi ride dataframe t."""

    scaled = t[num_vars].copy()

    scaled.iloc[:,:] = scaler.transform(scaled) # Convert to standard units

    categoricals = [pd.get_dummies(t[s], prefix=s, drop_first=True) for s in cat_vars]

    return pd.concat([scaled] + categoricals, axis=1)



design_matrix(train).iloc[0,:]
def rmse(errors):

    """Output: root mean squared error."""

    return np.sqrt(np.mean(errors ** 2))



constant_rmse = rmse(test['duration'] - np.mean(train['duration']))

constant_rmse
from sklearn.linear_model import LinearRegression



simple_model = LinearRegression()

simple_model = simple_model.fit(train[['distance']], train.loc[:, 'duration'])



simple_rmse = rmse(test['duration'] - simple_model.predict(test[['distance']]))

simple_rmse
linear_model = LinearRegression()

linear_model = linear_model.fit(design_matrix(train), train.loc[:, 'duration'])



linear_rmse = rmse(test['duration'] - linear_model.predict(design_matrix(test)))

linear_rmse
period_model = LinearRegression()

errors = []



for v in np.unique(train['period']):

    model = period_model.fit(design_matrix(train[train['period'] == v]), train[train['period'] == v].loc[:, 'duration'])

    errors = np.append(errors, test[test['period'] == v]['duration'] - period_model.predict(design_matrix(test[test['period'] == v])))



period_rmse = rmse(np.array(errors))

period_rmse
speed_model = LinearRegression()

speed_model = model.fit(design_matrix(train), train['speed'])



# Speed in miles/hr. Duration is measured in seconds, and there are 3600 seconds in an hour.

avg = (test['distance'] * 3600) / (speed_model.predict(design_matrix(test)))



speed_rmse = rmse(test['duration'] - avg)

speed_rmse
speed_model.predict(design_matrix(train))
speed_model.score(design_matrix(train), train['duration'])
tree_speed_model = LinearRegression()

choices = ['period', 'region', 'weekend']



def duration_error(predictions, observations):

    """Error between predictions (array) and observations (data frame)"""

    return predictions - observations['duration']



def speed_error(predictions, observations):

    """Duration error between speed predictions and duration observations"""

    return duration_error(observations['distance'] * 3600 / predictions, observations)



def tree_regression_errors(outcome='duration', error_fn=duration_error):

    """Return errors for all examples in test using a tree regression model."""

    errors = []

    for vs in train.groupby(choices).size().index:

        v_train, v_test = train, test

        for v, c in zip(vs, choices):

            v_train = v_train[v_train[c]==v]

            v_test = v_test[v_test[c]==v]

            print(v_train.shape, v_test.shape)

        tree_speed_model.fit(design_matrix(v_train), v_train.loc[:, outcome])

        errors = np.append(errors, error_fn(tree_speed_model.predict(design_matrix(v_test)), v_test))

    return errors



errors = tree_regression_errors()

errors_via_speed = tree_regression_errors('speed', speed_error)

tree_rmse = rmse(np.array(errors))

tree_speed_rmse = rmse(np.array(errors_via_speed))

print('Duration:', tree_rmse, '\nSpeed:', tree_speed_rmse)
models = ['constant', 'simple', 'linear', 'period', 'speed', 'tree', 'tree_speed']

pd.DataFrame.from_dict({

    'Model': models,

    'Test RMSE': [eval(m + '_rmse') for m in models]

}).set_index('Model').plot(kind='barh')

plt.xlabel('RMSE')

plt.title('RMSE by Model');