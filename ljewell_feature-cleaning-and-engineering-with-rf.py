import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Read 50,000 rows of data
train = pd.read_csv('../input/train.csv', nrows = 1000000)
test = pd.read_csv('../input/test.csv')
train.head(n=5)
print("Number of rows with zero values.")
print("Missing data in column", train[(train.isna())].shape[0])
print("Pickup longitude", train[(train.pickup_longitude == 0.0)].shape[0])
print("Pickup latitude", train[(train.pickup_latitude == 0.0)].shape[0])
print("Dropoff longitude", train[(train.dropoff_longitude == 0.0)].shape[0])
print("Dropoff latitude", train[(train.dropoff_latitude == 0.0) ].shape[0])
# Drop out these missing data rows.
train = train.dropna()
# Drop zero value
train = train[(train.pickup_longitude != 0.0 ) & \
              (train.pickup_latitude != 0.0) & \
              (train.dropoff_longitude != 0.0) &\
              (train.dropoff_latitude != 0.0)]
train[(train.pickup_latitude < -73.0 ) & (train.pickup_latitude > -75.0) & 
      (train.pickup_longitude > 40.4) & (train.pickup_latitude < 41.0) &
      (train.dropoff_latitude < -73.0 ) & (train.dropoff_latitude > -75.0) &
      (train.dropoff_longitude > 40.4) & (train.dropoff_latitude < 41.0)].head()
print("Calculating how many rows have transposed latitude and longitude")
# Pickup & dropoff lat/lon transposed.
print("Number of pickup & dropoff lat/lon transposed:", 
      train[(train.pickup_latitude < -73.0 ) & (train.pickup_latitude > -75.0) & 
      (train.pickup_longitude > 40.4) & (train.pickup_latitude < 41.0) &
      (train.dropoff_latitude < -73.0 ) & (train.dropoff_latitude > -75.0) &
      (train.dropoff_longitude > 40.4) & (train.dropoff_latitude < 41.0)].shape[0])
# Pickup lat/lon transposed but dropoff ok
print("Number of pickup lat/lon transposed but dropoff ok:", 
train[(train.pickup_latitude < -73.0 ) & (train.pickup_latitude > -75.0) & 
      (train.pickup_longitude > 40.4) & (train.pickup_latitude < 41.0) &
      (train.dropoff_latitude > -73.0 ) & (train.dropoff_latitude < -75.0) &
      (train.dropoff_longitude < 40.4) & (train.dropoff_latitude > 41.0)].shape[0])
# Pickup lat/lon ok but dropoff transposed
print("Number of pickup lat/lon ok but dropoff transposed:", 
train[(train.pickup_latitude > -73.0 ) & (train.pickup_latitude < -75.0) & 
      (train.pickup_longitude < 40.4) & (train.pickup_latitude > 41.0) &
      (train.dropoff_latitude > -73.0 ) & (train.dropoff_latitude < -75.0) &
      (train.dropoff_longitude < 40.4) & (train.dropoff_latitude > 41.0)].shape[0])
# Code from https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration kernek by Albert van Breemen.

import matplotlib.pyplot as plt
%matplotlib inline

# Grab out the rows which have transposed values
df_swap = train[(train.pickup_latitude < -73.0 ) & (train.pickup_latitude > -75.0) & 
      (train.pickup_longitude > 40.4) & (train.pickup_latitude < 41.0) &
      (train.dropoff_latitude < -73.0 ) & (train.dropoff_latitude > -75.0) &
      (train.dropoff_longitude > 40.4) & (train.dropoff_latitude < 41.0)]

nyc_map = plt.imread('https://aiblog.nl/download/nyc_-75_40_-73_41.5.png')
BB = (-74.25, -73.5, 40.4, 41.0)

# Plot them on the map to have a look at them
fig, axs = plt.subplots(1,1, figsize=(16,10))
axs.scatter(df_swap.pickup_latitude, df_swap.pickup_longitude, zorder=1, c='r')
axs.scatter(df_swap.dropoff_latitude, df_swap.dropoff_longitude, zorder=1, c='b')
axs.plot([df_swap.pickup_latitude, df_swap.dropoff_latitude], [df_swap.pickup_longitude, df_swap.dropoff_longitude], 'k-', lw=1)
axs.set_xlim((BB[0], BB[1]))
axs.set_ylim((BB[2], BB[3]))
axs.set_title('Pickup locations')
axs.imshow(nyc_map, zorder=0, extent=[-74.25, -73.5, 40.4, 41.0]);
train['pickup_longitude'] = train['pickup_longitude'].where(train.pickup_longitude < 39 , train['pickup_latitude'])
train['pickup_latitude'] = train['pickup_latitude'].where(train.pickup_latitude > -73 , train['pickup_longitude'])
train['dropoff_longitude'] = train['dropoff_longitude'].where(train.dropoff_longitude < 39 , train['dropoff_latitude'])
train['dropoff_latitude'] = train['dropoff_latitude'].where(train.dropoff_latitude > -73 , train['dropoff_longitude'])
#
print("Check that values have been swapped: ", train[(train.pickup_latitude < -73.0 ) & (train.pickup_latitude > -75.0) & 
      (train.pickup_longitude > 40.4) & (train.pickup_latitude < 41.0) &
      (train.dropoff_latitude < -73.0 ) & (train.dropoff_latitude > -75.0) &
      (train.dropoff_longitude > 40.4) & (train.dropoff_latitude < 41.0)].shape[0])
print("Number of rows where lat/lon doesn't change: ", train[(train.pickup_longitude == train.dropoff_longitude) & \
              (train.pickup_latitude == train.dropoff_latitude)].shape[0])
train[(train.pickup_longitude == train.dropoff_longitude) & \
              (train.pickup_latitude == train.dropoff_latitude)].head()
train = train[(train.pickup_longitude != train.dropoff_longitude) & \
              (train.pickup_latitude != train.dropoff_latitude)]
start = train.shape[0]
train = train[(train.pickup_longitude <= -73.0 ) & \
              (train.pickup_longitude >= -75.0) & \
              (train.pickup_latitude >= 40.4) & \
              (train.dropoff_latitude <= 41.0) & \
              (train.dropoff_longitude <= -73.0 ) & \
              (train.dropoff_longitude >= -75.0) & \
              (train.dropoff_latitude >= 40.4) & \
              (train.dropoff_latitude <= 41.0)]
print("Dropped {} with unreasonable lat or lon".format(start - train.shape[0]))
start = train.shape[0]
from geopy import distance
# Calculate distance function
def calc_dist(x):
    try:
        dist = round(distance.distance((x[0], x[1]),(x[2], x[3])).feet, 0)
    except ValueError as e:
        print("ERROR ", x, e)
        dist = 0
    return dist

train['distance'] = train[["pickup_latitude", "pickup_longitude", "dropoff_latitude","dropoff_longitude"]].apply(calc_dist, axis=1)
# Make sure distance and fare are resonable, fare should always be bigger than a straight line distance
before = train.shape[0]
train = train[((train.distance/1056)*0.50 < train.fare_amount)]
print("Dropping {} rows due to fare less than resonable amount according to distance".format(before-train.shape[0]))
train.head()
# Create feature for test set
test['distance'] = test[["pickup_latitude", "pickup_longitude", "dropoff_latitude","dropoff_longitude"]].apply(calc_dist, axis=1)
from geopy import distance

nyc_lat = 40.730610
nyc_lon = -73.935242
lon_step = 0.003
lat_step = 0.01

# If the lon moves by 0.03 how far is that
print("Longitude increasing by {} is {} miles".format(lon_step, round(distance.distance((nyc_lon, lat_step),(nyc_lon+lon_step, lat_step)).miles, 2)))
# If the lat moves by 0.03 how far is that
print("Latitude increasing by {} is {} miles".format(lat_step, round(distance.distance((nyc_lon, lat_step),(nyc_lon, lat_step+lat_step)).miles, 2)))
def bin_data(df):
    pickup_groups = df.groupby(["pickup_latitude", "pickup_longitude"])
    dropoff_groups = df.groupby(["dropoff_latitude", "dropoff_longitude"])
    trip_groups = df.groupby(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])
    print("Before bin:")
    print("Number of pickup locations: ",pickup_groups.ngroups)
    print("Number of dropoff locations: ",dropoff_groups.ngroups)
    print("Number of trips: ", trip_groups.ngroups)
    #
    # Create the bin size.
    lon_step = 0.003
    lat_step = 0.01
    print("Lat bin size {} and long bin size {}\n".format(lat_step, lon_step))
    to_bin_lon = lambda x: np.floor(x / lon_step) * lon_step
    to_bin_lat = lambda x: np.floor(x / lat_step) * lat_step
    df["pickup_latbin"] = df.pickup_latitude.map(to_bin_lat)
    df["pickup_lonbin"] = df.pickup_longitude.map(to_bin_lon)
    df["dropoff_latbin"] = df.dropoff_latitude.map(to_bin_lat)
    df["dropoff_lonbin"] = df.dropoff_longitude.map(to_bin_lon)
    pickup_groups = df.groupby(["pickup_latbin", "pickup_lonbin"])
    dropoff_groups = df.groupby(["dropoff_latbin", "dropoff_lonbin"])
    trip_groups = df.groupby(["pickup_latbin", "pickup_lonbin","dropoff_latbin", "dropoff_lonbin"])
    print("After bin:")
    print("New number of pickup locations: ",pickup_groups.ngroups)
    print("New number of dropoff locations: ",dropoff_groups.ngroups)
    print("New number of trips: ", trip_groups.ngroups)

    return df
train = bin_data(train)
test = bin_data(test)
train.head()
# We can drop off the columns
train = train.drop(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], axis=1)
test = test.drop(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], axis=1)
def convert_date(df):
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime, format='%Y-%m-%d %H:%M:%S %Z', utc=True)
    # Add New York timezone
    df['pickup_datetime'] = df.pickup_datetime.dt.tz_convert('America/New_York')
    df['year'] = df.pickup_datetime.dt.year
    df['pickup_hour'] = df.pickup_datetime.dt.hour
    # day of week 0 Monday to 6 Sunday, we can capture weekend with < 5
    df['weekend'] = df.pickup_datetime.dt.dayofweek < 5
    # Add in surcharge period
    df['surcharge'] = ((df.pickup_hour >= 20) | (df.pickup_hour <= 5))
    # Drop pickup_datetime as we have extracted everything we need
    df = df.drop('pickup_datetime', axis=1)

    return df
train = convert_date(train)
test = convert_date(test)
train.head()
test.head()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

model = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("rf_tree", RandomForestRegressor())
    ))

X = train.iloc[:,2:].astype('float64', axis=1)
y = np.ravel(train.fare_amount.values)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
model.fit(X=train_X, y=train_y)
predict_y = model.predict(X=test_X)
rmse = np.sqrt(mean_squared_error(test_y, predict_y))
compare = pd.DataFrame()
compare['train_y'] = test_y
compare['predict_y'] = predict_y
compare['RSE'] = np.sqrt(np.square(test_y - predict_y))
print("RMSE: ", rmse)
compare.head(n=10)
# Cross validate to see how robust the results are
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Cross validated RMSE:", np.sqrt(np.mean(scores)*-1))
test_X = test.iloc[:,1:].astype('float64', axis=1)

model.fit(X=X, y=y)
y_pred_final = model.predict(test_X)

submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
submission.head(n=5)