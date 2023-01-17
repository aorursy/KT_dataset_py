# Import basic required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.style.use('bmh')
sns.set_style({'axes.grid':False}) 

# Advanced visualization modules(datashader)
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis, inferno

# Folium visualization for geographical map
import folium as flm
from folium.plugins import HeatMap
%%time
# Downcasting data types to reduce momory consumption
dtypes = {}
for key in ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']:
    dtypes[key] = 'float32'
for key in ['passenger_count']:
    dtypes[key] = 'uint8'
    
# Read in train and test data (5 million rows)
train = pd.read_csv('../input/train.csv', nrows = 5_000_000, dtype = dtypes).drop('key', axis = 1)
test = pd.read_csv('../input/test.csv', dtype = dtypes)
# Now check out the data types 
print('Dtypes after downcasting except pickup_datetime:')
display(train.dtypes)
%%time
# 'pickup_datetime' should be in datetime format. Let's convert it
# Don't forget to set 'infer_datetime_format=True'. Otherwise it takes forever :)
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True)
# Current memory usage(in MB) by columns after conversion
print('Memory usage(in MB) by variables after conversion:')
display(np.round(train.memory_usage(deep = True)/1024**2, 4))
# Look at the data we're going to deal with
print('Preview train data:')
display(train.head())
print('Preview test data:')
display(test.head())
# Missing values in train data
print('Missing values in train data:')
display(train.isna().sum())

# Missing values in test data
print('Missing values in test data:')
display(test.isna().sum())
# Drop missing observations from train data.
train.dropna(how = 'any', axis = 0, inplace = True)
# Shape of the df after dropping missing rows
print('Shape of the df after dropping missing rows:{}'.format(train.shape))
# Distrubution of target variable with skewness
fig, ax = plt.subplots(figsize = (14,6))
sns.distplot(train.fare_amount, bins = 200, color = 'firebrick', ax = ax)
ax.set_title('Distribution of fare_amount (skewness: {:0.5})'.format(train.fare_amount.skew()))
ax.set_ylabel('realtive frequency')
plt.show()
# Class distribution of passenger_count
fig, ax = plt.subplots(figsize = (14,6))
class_dist = train.passenger_count.value_counts()
class_dist.plot(kind = 'bar', ax = ax)
ax.set_title('Class distribution of passenger_count')
ax.set_ylabel('absolute frequency')
plt.show()
# Look at the abnormalities using descritive stats
train.fare_amount.describe()
# Drop fare_amount less than 0.
neg_fare = train.loc[train.fare_amount<0, :].index
train.drop(neg_fare, axis = 0, inplace = True)

# Rerun the descriptive stats
train.fare_amount.describe()
# Drop rows greater than 100 and lesser than 2.5
fares_to_drop = train.loc[(train.fare_amount>100) | (train.fare_amount<2.5), :].index
train.drop(fares_to_drop, axis = 0, inplace = True)
print('Shape of train data after dropping outliers from fare_amount:{}'.format(train.shape))
# Check the 2.5 and 97.5 percentile of los nad lats
def percentile(variable):
    two_and_half = variable.quantile(0.25)
    ninty_seven_half = variable.quantile(0.975)
    print('2.5 and 97.5 percentile of {} is respectively: {:0.2f}, and {:0.2f}'.format(variable.name, two_and_half, ninty_seven_half))
    
percentile(train.pickup_latitude)
percentile(train.dropoff_latitude)
percentile(train.pickup_longitude)
percentile(train.dropoff_longitude) 
# For lats, our range is 40 to 42 degrees(with 40 and 42)
train = train.loc[train.pickup_latitude.between(left = 40, right = 42), :]
train = train.loc[train.dropoff_latitude.between(left = 40, right = 42), :]

# For lons, our range is -75 to -72 degrees(with 40 and 42)
train = train.loc[train.pickup_longitude.between(left = -75, right = -72), :]
train = train.loc[train.dropoff_longitude.between(left = -75, right = -72), :]
print('Shape of train data after after dropping outliers from lats and lons: {}'.format(train.shape))
# Check out the descriptive stats first
train.passenger_count.describe()
# Drop passenger_count of 208 and 129, 9, and 7.
passenger_count_to_drop = train.loc[(train.passenger_count==208) | (train.passenger_count==129) | (train.passenger_count==9) | (train.passenger_count==7)].index
train.drop(passenger_count_to_drop, axis = 0, inplace = True)
print('Shape of train data after dropping outliers from passenger_count:{}'.format(train.shape))

# Let's check again the passenger_count
display(train.passenger_count.describe())
# Merged train and test data across rows
merged = pd.concat([train,test], axis = 0, sort=False)
# Calculate great circle distance using haversine formula
def great_circle_distance(lon1,lat1,lon2,lat2):
    R = 6371000 # Approximate mean radius of earth (in m)
    
    # Convert decimal degrees to ridians
    lon1,lat1,lon2,lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Distance of lons and lats in radians
    dis_lon = lon2 - lon1
    dis_lat = lat2 - lat1
    
    # Haversine implementation
    a = np.sin(dis_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dis_lon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dis_m = R*c # Distance in meters
    dis_km = dis_m/1000 # Distance in km
    return dis_km

# Create a column named greate_circle_distance
merged['great_circle_distance'] = great_circle_distance(merged.pickup_longitude, merged.pickup_latitude, merged.dropoff_longitude, merged.dropoff_latitude)
# Convert lons and lats into cartesian coordinates. Assume the earth as sphere not ellipsoid
R = 6371000 # Approximate mean radius of earth (in m)
 # lons and lats must be in radians
lon1,lat1,lon2,lat2 = map(np.radians, [merged.pickup_longitude, merged.pickup_latitude, merged.dropoff_longitude, merged.dropoff_latitude])
merged['pickup_x'] = R*np.cos(lon1)*np.cos(lat1)
merged['pickup_y'] = R*np.sin(lon1)*np.cos(lat1)
merged['dropoff_x'] = R*np.cos(lon2)*np.cos(lat2)
merged['dropoff_y'] = R*np.sin(lon2)*np.cos(lat2)

# Now calculate the euclidean distance
x1 = merged['pickup_x']
y1 = merged['pickup_y']
x2 = merged['dropoff_x']
y2 = merged['dropoff_y']
merged['euclidean_distance'] = (np.sqrt(( x1 - x2)**2 + ( y1 - y2)**2))/1000 # in km
# Calculate manhattan distance from x and y coordinates
merged['manhattan_distance'] = (np.abs(x1 - x2) + np.abs(y1 - y2))/1000 # in km
# Create two variables taking absolute differences of lons and lats
merged['abs_lon_diff'] = np.abs(merged.pickup_longitude - merged.dropoff_longitude)
merged['abs_lat_diff'] = np.abs(merged.pickup_latitude - merged.dropoff_latitude)
# Extract pickup_hour, day, date, month, and year from pickup_datetime.
merged['pickup_hour'] = merged.pickup_datetime.dt.hour
merged['pickup_date'] =  merged.pickup_datetime.dt.day
merged['pickup_day_of_week'] =  merged.pickup_datetime.dt.dayofweek
merged['pickup_month'] =  merged.pickup_datetime.dt.month
merged['pickup_year'] =  merged.pickup_datetime.dt.year
# Let's see the current dtypes and total memory consumption by variables in MB
print('Current Data Types:')
display(merged.dtypes)
print('\n Total memory consumption in MB: {}'.format(np.sum(merged.memory_usage(deep = True)/1024**2)))
# Drop variables 
merged.drop(['key', 'pickup_datetime'], axis = 1, inplace = True)
# Downcasting variables
merged.loc[:, ['pickup_hour', 'pickup_date', 'pickup_day_of_week', 'pickup_month']] = merged.loc[:, ['pickup_hour', 'pickup_date', 'pickup_day_of_week', 'pickup_month']].astype(np.uint8)
merged.loc[:, ['great_circle_distance', 'euclidean_distance', 'manhattan_distance']] = merged.loc[:, ['great_circle_distance', 'euclidean_distance', 'manhattan_distance']].astype(np.float32)
merged.loc[:, ['pickup_year']] = merged.loc[:, ['pickup_year']].astype('int16')

# Check total memory consumption after downcasting
print('Total memory consumption after downcasting in MB: {}'.format(np.sum(merged.memory_usage(deep = True)/1024**2)))
# Let's separate train and test data again
train_df = merged.iloc[0:4892576, :]
test_df = merged.iloc[4892576:, :] 
test_df.drop('fare_amount', axis = 1, inplace = True) # Due to concatenation
# Let's see which variables have the strongest and weakest correlation with fare_amount
corr = train_df.corr().sort_values(by='fare_amount', ascending=False)
fig, ax = plt.subplots(figsize = (20,12))
sns.heatmap(corr, annot = True, cmap ='BrBG', ax = ax, fmt='.2f', linewidths = 0.05, annot_kws = {'size': 17})
ax.tick_params(labelsize = 15)
ax.set_title('Correlation with fare_amount', fontsize = 22)
plt.show()
# Plot subplots of regression plots
continuous_var = train_df.iloc[0:10000, :].select_dtypes(include = ['float32', 'float64']).drop('fare_amount', axis = 1)
fig, axes = plt.subplots(7,2, figsize = (40,80))
for ax, column in zip(axes.flatten(), continuous_var.columns):
    x = continuous_var[column]
    y = train_df.fare_amount.iloc[0:10000]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    sns.regplot(x = continuous_var[column], y = y, ax = ax, line_kws={'label':'r: {}\np: {}'.format(r_value,p_value)})
    ax.set_title('{} vs fare_amount'.format(column), fontsize = 36)
    fig.suptitle('Regression Plots', fontsize = 45)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 22)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 22)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.legend(loc = 'best', fontsize = 32)
fig.delaxes(ax = axes[6,1])
fig.tight_layout(rect = [0, 0.03, 1, 0.97])
# Extract categorical variable first
cat_var = train_df.iloc[0:10000, :].select_dtypes(include = ['uint8'])
cat_var = pd.concat([cat_var, train_df.pickup_year.iloc[0:10000]], axis = 1)

# A box plot to visualize the association between fare_amount and categorical variables
fig, axes = plt.subplots(3,2,figsize = (20,25))
for ax, column in zip(axes.flatten(), cat_var.columns):
    sns.boxplot(x = cat_var[column], y = train_df.fare_amount.iloc[0:10000], ax = ax)
    ax.set_title('{} vs fare_amount'.format(column), fontsize = 22)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
    ax.set_xlabel(column, fontsize = 18)
    ax.set_ylabel('fare_amount', fontsize = 18)
    fig.suptitle('Association with categorical variables', fontsize = 26)
fig.tight_layout(rect = [0, 0.03, 1, 0.97])
# Let's group mean fare_amount by pickup_year to see if there is a pattern.
pivot_year = pd.pivot_table(train_df, values = 'fare_amount', index = 'pickup_year', aggfunc = ['mean'])
print('Mean fare_amount across the classes of pickup_year: \n{}'.format(pivot_year))
# or train.fare_amount.groupby([train.pickup_year]).mean()
# A bar plot would be more helpful to visualize this pattern
fig, ax = plt.subplots(figsize = (15,5))
pivot_year.plot(kind = 'bar', legend = False, color = 'firebrick', ax = ax)
ax.set(title = 'pickup_year vs mean fare_amount', ylabel= 'mean fare_amount')
plt.show()
# x_range and y_range for pickup_locations
print('x_range and y_range for pickup_locations:')
print(train_df.pickup_longitude.min(), train_df.pickup_longitude.max())
print(train_df.pickup_latitude.min(), train_df.pickup_latitude.max())

# x_range and y_range for dropoff_locations
print('\nx_range and y_range for dropoff_locations:')
print(train_df.dropoff_longitude.min(), train_df.dropoff_longitude.max())
print(train_df.dropoff_latitude.min(), train_df.dropoff_latitude.max())
# Create a function to plot longitudes vs latitudes of rides
def plot_location(lon,lat, c_map):
    # Initial datashader visualization configuration
    pickup_range = dropoff_range = x_range, y_range = ((-74.05, -73.7), (40.6, 40.85))
    # Initiate canvas and create grid
    cvs = ds.Canvas(plot_width = 1080, plot_height = 600, x_range = x_range, y_range = y_range)
    agg = cvs.points(train, lon, lat)
    # Create image map with custom color map
    img = tf.shade(agg, cmap = c_map, how = 'eq_hist')
    return tf.set_background(img, 'black')
# Show image map of pickup locations with viridis color map
plot_location('pickup_longitude', 'pickup_latitude', viridis)
# Show image map of pickup locations with inferno color map
plot_location('pickup_longitude', 'pickup_latitude', inferno)
# Create a function to plot folium heatmap
def plot_map(lat, lon):
    # Lat and lon of nyc to plot the map of nyc
    map_nyc = flm.Map(location = [40.7141667, -74.0063889], zoom_start = 12, tiles = "Stamen Toner")
    # creates a marker for nyc
    flm.Marker(location = [40.7141667, -74.0063889], icon = flm.Icon(color = 'red'), popup='NYC').add_to(map_nyc)
    # Plot heatmap of 20000 lats and lons points
    lat_lon = train.loc[0:20000, [lat, lon]].values
    HeatMap(lat_lon, radius = 10).add_to(map_nyc)
    #map_nyc.save('HeatMap.html')
    return map_nyc
# Plot street map of NYC and then plot heatmap of pickup locations on it.
plot_map('pickup_latitude', 'pickup_longitude')
# Show image map of pickup locations with inferno color map
plot_location('dropoff_longitude', 'dropoff_latitude', inferno)
# viridis color map is even better to capture patterns
plot_location('dropoff_longitude', 'dropoff_latitude', viridis)
# Plot street map of NYC and then plot heatmap of dropoffs lats and lons on it.
plot_map('dropoff_latitude', 'dropoff_longitude')
# Get the data ready for training and predicting
y_train = train_df.fare_amount
X_train = train_df.drop(['fare_amount'], axis = 1)
X_test = test_df
# Train and predict using linear regression
from sklearn.linear_model import LinearRegression

# Instantiate linear regression object
linear_reg = LinearRegression()

# Train with the objt
linear_reg.fit(X_train, y_train)

# Make prediction
y_pred = linear_reg.predict(X_test)
# Create csv file for submission
submission = pd.DataFrame()
submission['key'] = test.key
submission['fare_amount'] = y_pred
submission.to_csv('sub_with_linear_reg', index = False)