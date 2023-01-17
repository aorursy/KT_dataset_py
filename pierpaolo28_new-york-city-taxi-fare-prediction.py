import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 500_000, 

                   parse_dates = ['pickup_datetime']).drop(columns = 'key')



# Remove na

df = df.dropna()

df.head()
df.describe()
plt.figure(figsize = (10,8))

plt.hist(df['fare_amount'])

plt.title('Fare Distribution')
print(f"Number of negative fares: {len(df[df['fare_amount'] < 0])}")

print(f"Number of fares equal to 0: {len(df[df['fare_amount'] == 0])}")
df = df[df['fare_amount'].between(left = 2.5, right= df['fare_amount'].max())]
def ecdf(x):

    x = np.sort(x)

    n = len(x)

    # Going from 1/n to 1

    y = np.arange(1, n + 1, 1) / n

    return x, y
x, y = ecdf(df['fare_amount'])

plt.figure(figsize = (8, 6))

plt.plot(x, y)

plt.ylabel('Percentile'); 

plt.xlabel('Fare Amount');

plt.title('Fare Amount ECDF'); 
df = df[df['fare_amount'].between(left = 2.5, right= 70)]
x, y = ecdf(df['fare_amount'])

plt.figure(figsize = (8, 6))

plt.plot(x, y)

plt.ylabel('Percentile'); 

plt.xlabel('Fare Amount');

plt.title('Fare Amount ECDF'); 
df['passenger_count'].value_counts().plot.bar();

plt.title('Passenger Counts')

plt.xlabel('Passengers Numbers') 

plt.ylabel('Frequency')
df = df.loc[df['passenger_count'] < 6]
fig, axes = plt.subplots(1, 2, figsize = (20, 8), sharex=True, sharey=True)

axes = axes.flatten()



# Plot Longitude (x) and Latitude (y)

sns.regplot('pickup_longitude', 'pickup_latitude', fit_reg = False, 

            data = df, ax = axes[0]);

sns.regplot('dropoff_longitude', 'dropoff_latitude', fit_reg = False, 

            data = df, ax = axes[1]);

axes[0].set_title('Pickup Locations')

axes[1].set_title('Dropoff Locations');
# Absolute difference in latitude and longitude

df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()

df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()
sns.lmplot('abs_lat_diff', 'abs_lon_diff', fit_reg = False, data = df)

plt.title('Absolute latitude difference vs Absolute longitude difference')
zero_diff = df[(df['abs_lat_diff'] == 0) & (df['abs_lon_diff'] == 0)]

zero_diff.shape
def minkowski_distance(x1, x2, y1, y2, p):

    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
df['euclidean'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],

                                       df['pickup_latitude'], df['dropoff_latitude'], 2)
plt.figure(figsize = (10,8))

plt.hist(df['euclidean'])

plt.title('Euclidean Distance Distribution')

ax = plt.subplot(111)

ax.set_xlim([0, 500])
plt.figure(figsize = (10, 6))



for p, grouped in df.groupby('passenger_count'):

    sns.kdeplot(grouped['fare_amount'], label = f'{p} passengers');

    

plt.xlabel('Fare Amount'); plt.ylabel('Density')

plt.title('Distribution of Fare Amount by Number of Passengers');
df.groupby('passenger_count')['fare_amount'].agg(['mean', 'count'])
df.groupby('passenger_count')['fare_amount'].mean().plot.bar(color = 'b');

plt.title('Average Fare by Passenger Count');
# Radius of the earth in kilometers

R = 6378



def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    

    

    source: https://stackoverflow.com/a/29546836



    """

    # Convert latitude and longitude to radians

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    # Find the differences

    dlon = lon2 - lon1

    dlat = lat2 - lat1



    # Apply the formula 

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    # Calculate the angle (in radians)

    c = 2 * np.arcsin(np.sqrt(a))

    # Convert to kilometers

    km = R * c

    

    return km
df['haversine'] =  haversine_np(df['pickup_longitude'], df['pickup_latitude'],

                         df['dropoff_longitude'], df['dropoff_latitude']) 
sns.kdeplot(df['haversine']);
corrs = df.corr()

corrs['fare_amount'].plot.bar(color = 'b');

plt.title('Correlation with Fare Amount');
test = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv', 

                   parse_dates = ['pickup_datetime'])



# Create absolute differences

test['abs_lat_diff'] = (test['dropoff_latitude'] - test['pickup_latitude']).abs()

test['abs_lon_diff'] = (test['dropoff_longitude'] - test['pickup_longitude']).abs()



# Save the id for submission

test_id = list(test.pop('key'))



test['euclidean'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],

                                       test['pickup_latitude'], test['dropoff_latitude'], 2)



test['haversine'] = haversine_np(test['pickup_longitude'], test['pickup_latitude'],

                         test['dropoff_longitude'], test['dropoff_latitude'])



test.describe()
from sklearn.model_selection import train_test_split



# Split data

X_train, X_valid, y_train, y_valid = train_test_split(df, np.array(df['fare_amount']), 

                                                      test_size = 0.30)
import xgboost as xgb



xgbr = xgb.XGBRegressor()

xgbr.fit(X_train[['haversine', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count']], y_train)
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)



def metrics(train_pred, valid_pred, y_train, y_valid):

    """Calculate metrics:

       Root mean squared error and mean absolute percentage error"""

    

    # Root mean squared error

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    

    # Calculate absolute percentage error

    train_ape = abs((y_train - train_pred) / y_train)

    valid_ape = abs((y_valid - valid_pred) / y_valid)

    

    # Account for y values of 0

    train_ape[train_ape == np.inf] = 0

    train_ape[train_ape == -np.inf] = 0

    valid_ape[valid_ape == np.inf] = 0

    valid_ape[valid_ape == -np.inf] = 0

    

    train_mape = 100 * np.mean(train_ape)

    valid_mape = 100 * np.mean(valid_ape)

    

    return train_rmse, valid_rmse, train_mape, valid_mape



def evaluate(model, features, X_train, X_valid, y_train, y_valid):

    """Mean absolute percentage error"""

    

    # Make predictions

    train_pred = model.predict(X_train[features])

    valid_pred = model.predict(X_valid[features])

    

    # Get metrics

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,

                                                             y_train, y_valid)

    

    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')

    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')
evaluate(xgbr, ['haversine', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count'],

         X_train, X_valid, y_train, y_valid)
preds = xgbr.predict(test[['haversine', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count']])



sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})

sub.to_csv('sub_rf_simple.csv', index = False)



sns.distplot(sub['fare_amount'])

plt.title('Distribution of Random Forest Predicted Fare Amount');