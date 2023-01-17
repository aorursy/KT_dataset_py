# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
# Pandas display options
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Set random seed 
RSEED = 2020
# Visualizations
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 12
import seaborn as sns
palette = sns.color_palette('Paired', 10)
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingCVRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
data = pd.read_csv('../input/train.csv', nrows = 5_000_00, 
                   parse_dates = ['pickup_datetime']).drop(columns = 'key')

# Remove na
data = data.dropna()
data.head()
data.describe()
data.isna().sum()/data.shape[0]*100
#distribution of target (fare amount)
sns.distplot(data['fare_amount'])
def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    # Sort in ascending order
    x = np.sort(x)
    n = len(x)
    # Go from 1/n to 1
    y = np.arange(1, n + 1, 1) / n
    return x, y
xs, ys = ecdf(data['fare_amount'])
plt.figure(figsize = (8, 6))
plt.plot(xs, ys, '.')
plt.ylabel('Percentile'); plt.title('ECDF of Fare Amount'); plt.xlabel('Fare Amount ($)');
data['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count');
print('ada '+str(data[data['passenger_count']==0].shape[0])+'transaksi dengan 0 passangger')
print('ada '+str(data[data['passenger_count']==6].shape[0])+'transaksi dengan 6 passangger')
data = data.loc[data['pickup_latitude'].between(40, 42)]
data = data.loc[data['pickup_longitude'].between(-75, -72)]
data = data.loc[data['dropoff_latitude'].between(40, 42)]
data = data.loc[data['dropoff_longitude'].between(-75, -72)]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
cluster_pickup = kmeans.fit_predict(data[['pickup_longitude','pickup_latitude']])
cluster_dropoff = kmeans.fit_predict(data[['dropoff_longitude','dropoff_latitude']])
data['cluster_pickup']=cluster_pickup
data['cluster_dropoff']=cluster_dropoff
# this function will also be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

# load extra image to zoom in on NYC
BB_zoom = (-74.1, -73.7, 40.6, 40.85)
nyc_map_zoom = plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true')
# this function will be used more often to plot data on the NYC map
def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2, color = False):
    fig, axs = plt.subplots(1, 2, figsize=(18, 22))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title('Pickup locations')
    axs[0].axis('off')
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='b', s=s)
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].axis('off')
    axs[1].imshow(nyc_map, zorder=0, extent=BB)
    
# plot training data on map zoomed in
plot_on_map(data.sample(4_000_00, random_state = RSEED), 
            BB_zoom, nyc_map_zoom, s=0.05, alpha=0.05)
# Create a color mapping based on fare bins
color_mapping = {cluster_pickup: palette[i] for i, cluster_pickup in enumerate(data['cluster_pickup'].unique())}
data['color'] = data['cluster_pickup'].map(color_mapping)
plot_data = data.sample(4_000_00, random_state = RSEED)
BB = BB_zoom

fig, axs = plt.subplots(1, 1, figsize=(20, 18))

# Plot the pickups
for b, df in plot_data.groupby('cluster_pickup'):
    # Set the zorder to 1 to plot on top of map
    axs.scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=0.2, c=df.color, s=30, label = f'{b}')
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('Pickup locations', size = 32)
    axs.axis('off')
    
# Legend
leg = axs.legend(fontsize = 28, markerscale = 3)

# Adjust alpha of legend markers
for lh in leg.legendHandles: 
    lh.set_alpha(1)

leg.set_title('Cluster Pickup', prop = {'size': 28})

# Show map in background (zorder = 0)
axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);

# Create a color mapping based on fare bins
color_mapping = {cluster_pickup: palette[i] for i, cluster_pickup in enumerate(data['cluster_dropoff'].unique())}
data['color'] = data['cluster_dropoff'].map(color_mapping)
plot_data = data.sample(4_000_00, random_state = RSEED)
fig, axs = plt.subplots(1, 1, figsize=(20, 18))

# Plot the pickups
for b, df in plot_data.groupby('cluster_dropoff'):
    axs.scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, 
                alpha=0.2, c=df.color, s=30, label = f'{b}')
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('cluster_dropoff', size = 32)
    axs.axis('off')
    
# Legend
leg = axs.legend(fontsize = 28, markerscale = 3)

# Adjust alpha of legend markers
for lh in leg.legendHandles: 
    lh.set_alpha(1)

leg.set_title('Cluster Dropoff', prop = {'size': 28})

# Show map in background (zorder = 0)
axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
                                                           
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
                                                           
                                                           
place = pd.DataFrame({'loc' : ['jfk','nyc','ewr','lgr'], 'long' : [-73.7822222222,-74.0063889,-74.175,-73.87], 'lat' : [40.6441666667,40.7141667,40.69,40.77]})

def distance_to_place(df,location,source_long,source_lat):
    selected_place = place[place['loc']==location]
    selected_place = selected_place.reset_index()
    xx = haversine_np(df[source_long], df[source_lat], selected_place['long'][0], selected_place['lat'][0])
    
    return xx
                                                           

def calculate_direction(df):
    d_lon = df['pickup_longitude'] - df['dropoff_longitude']
    d_lat = df['pickup_latitude'] - df['dropoff_latitude']
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result
data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()

data['manhattan'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'],
                                       data['pickup_latitude'], data['dropoff_latitude'], 1)

data['euclidean'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'],
                                       data['pickup_latitude'], data['dropoff_latitude'], 2)


data['haversine'] =  haversine_np(data['pickup_longitude'], data['pickup_latitude'],
                         data['dropoff_longitude'], data['dropoff_latitude']) 

for i in place['loc'].tolist():
    for j in ['pickup','dropoff']:
        data[str(j)+'_distance_to'+str(i)] = distance_to_place(data,i,str(j)+'_longitude',str(j)+'_latitude')

data['direction'] = calculate_direction(data)

data = extract_dateinfo(data, 'pickup_datetime', drop = False, 
                         time = True, start_ref = data['pickup_datetime'].min())
data['haversine'].describe()
cek = data[data['haversine']<200]
sns.distplot(cek['haversine'])
(data['haversine']>25).sum()
# scatter plot distance - fare
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(data.haversine, data.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance km')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')

# zoom in on part of data
idx = (data.haversine <= 25) & (data.fare_amount < 100)
axs[1].scatter(data[idx].haversine, data[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance km')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 15 km, fare < $100');
data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)].shape[0], data.shape[0], data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)].shape[0]/data.shape[0]
sns.distplot(data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)]['passenger_count'])
plt.show()
sns.distplot(data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)]['fare_amount'])
plt.show()
corrs = data.corr()
corrs = corrs.drop('fare_amount',axis=0)
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');
cek_cols = corrs.columns.tolist()
cek = data.copy()
for i in cek_cols:
    cek[i] = np.log(cek[i])
corrs = cek.corr()
corrs = corrs.drop('fare_amount',axis=0)
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');
#time
import re
def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(1900, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    # Extract the field
    fld = df[date_col]
    
    # Check the time
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    # Convert to datetime if not already
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    # Prefix for new columns
    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    # Basic attributes
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    # Additional attributes
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    # If time is specified, extract time information
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    # Iterate through each attribute
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    # Calculate days in year
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        # Add fractional time of day (0 - 1) units of day
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        # Add fractional time of week (0 - 1) units of week
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        # Add fractional time of month (0 - 1) units of month
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        # Add fractional time of year (0 - 1) units of year
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    # Add seconds since start of reference
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df

data = extract_dateinfo(data, 'pickup_datetime', drop = False,time = True, start_ref = df['pickup_datetime'].min())
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,12])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
time_slicer(df=data, timeframes=['pickup_Year', 'pickup_Month', 'pickup_Day','pickup_Hour'], value = "fare_amount", color="blue")
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
holidays = calendar().holidays()
data["usFedHoliday"] =  data.pickup_datetime.dt.date.astype('datetime64').isin(holidays)
cek =  data[(data.haversine <= 25) & (data.fare_amount <= 50) & (data.usFedHoliday == True)]
sns.distplot(cek['fare_amount'])
cek1 =  data[(data.haversine <= 25) & (data.fare_amount <= 50) & (data.usFedHoliday == False)]
sns.distplot(cek['fare_amount'])
data = pd.read_csv('../input/train.csv', nrows = 5_000_0, 
                   parse_dates = ['pickup_datetime']).drop(columns = 'key')

# Remove na
data = data.dropna()
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
                                                           
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
                                                           
                                                           
place = pd.DataFrame({'loc' : ['jfk','nyc','ewr','lgr'], 'long' : [-73.7822222222,-74.0063889,-74.175,-73.87], 'lat' : [40.6441666667,40.7141667,40.69,40.77]})

def distance_to_place(df,location,source_long,source_lat):
    selected_place = place[place['loc']==location]
    selected_place = selected_place.reset_index()
    xx = haversine_np(df[source_long], df[source_lat], selected_place['long'][0], selected_place['lat'][0])
    
    return xx
                                                           

def calculate_direction(df):
    d_lon = df['pickup_longitude'] - df['dropoff_longitude']
    d_lat = df['pickup_latitude'] - df['dropoff_latitude']
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result
#time
import re
def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(1900, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    # Extract the field
    fld = df[date_col]
    
    # Check the time
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    # Convert to datetime if not already
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    # Prefix for new columns
    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    # Basic attributes
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    # Additional attributes
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    # If time is specified, extract time information
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    # Iterate through each attribute
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    # Calculate days in year
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        # Add fractional time of day (0 - 1) units of day
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        # Add fractional time of week (0 - 1) units of week
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        # Add fractional time of month (0 - 1) units of month
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        # Add fractional time of year (0 - 1) units of year
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    # Add seconds since start of reference
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df
class data_transform():
    
    
    def __init__(self, num, cat, is_cat):
        self.num = num
        self.cat = cat
        self.is_cat = is_cat
        
    def fit(self, X):
        # do not do anything
        return self
    
    def transform(self, X, y = None):
        num_cols = self.num
        cat_cols = self.cat
        df = X.copy()
        df['passenger_count'] = np.where(df['passenger_count']<1,1,df['passenger_count'])
        df['passenger_count'] = np.where(df['passenger_count']>6,5,df['passenger_count'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']<40, 40,df['pickup_latitude'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']>42,42,df['pickup_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']<40, 40,df['dropoff_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']>42, 42,df['dropoff_latitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']<-75, -75,df['pickup_longitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']>-72, -72,df['pickup_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']<-75, -75,df['dropoff_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']>-72, -72,df['dropoff_longitude'])

        kmeans = KMeans(n_clusters=5)
        cluster_pickup = kmeans.fit_predict(data[['pickup_longitude','pickup_latitude']])
        cluster_dropoff = kmeans.fit_predict(data[['dropoff_longitude','dropoff_latitude']])
        data['cluster_pickup']=cluster_pickup
        data['cluster_dropoff']=cluster_dropoff
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        # Absolute difference in latitude and longitude
        df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
        df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()

        df['manhattan'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 1)

        df['euclidean'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 2)

        df['haversine'] =  haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                 df['dropoff_longitude'], df['dropoff_latitude']) 

        for i in place['loc'].tolist():
            for j in ['pickup','dropoff']:
                df[str(j)+'_distance_to'+str(i)] = distance_to_place(df,i,str(j)+'_longitude',str(j)+'_latitude')

        df['direction'] = calculate_direction(df)

        df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                                 time = True, start_ref = df['pickup_datetime'].min())
        
        holidays = calendar().holidays()
        data["usFedHoliday"] =  data.pickup_datetime.dt.date.astype('datetime64').isin(holidays)

        df[cat_cols] = df[cat_cols].astype(str)
        
        
        if self.is_cat==1:
            df = df[cat_cols]
        elif self.is_cat==0:
            df = df[num_cols] 
        else:
            df = df
            
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    result = np.sqrt(mean_squared_error(y_true, y_pred))
    return result

# def mape(y_true, y_pred):
#     result = np.abs(y_true-y_pred) / y_true
#     result = np.mean(result)
#     return result

# def my_scorer1():
#     return make_scorer(rmse, greater_is_better=False)
ori_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
num_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'abs_lat_diff', 'abs_lon_diff', 'manhattan', 'euclidean', 'haversine',
       'pickup_distance_tojfk', 'dropoff_distance_tojfk',
       'pickup_distance_tonyc', 'dropoff_distance_tonyc',
       'pickup_distance_toewr', 'dropoff_distance_toewr',
       'pickup_distance_tolgr', 'dropoff_distance_tolgr', 'direction',
       'pickup_Year', 'pickup_Month', 'pickup_Week', 'pickup_Day',
       'pickup_Dayofweek', 'pickup_Dayofyear', 'pickup_Days_in_month',
       'pickup_Hour', 'pickup_Minute', 'pickup_Second',
       'pickup_Days_in_year', 'pickup_frac_day', 'pickup_frac_week',
       'pickup_frac_month', 'pickup_frac_year', 'pickup_Elapsed']
cat_cols = ['pickup_is_leap_year','usFedHoliday']
target = 'fare_amount'
from sklearn.model_selection import train_test_split
train, val = train_test_split(data,test_size=0.2, random_state=RSEED)
del data
num_transformer = Pipeline(steps=[
                                ('dataprep', data_transform(num_cols,cat_cols,is_cat=0)),
                                ('imputer', SimpleImputer(strategy = "mean")),
                                ('scaler', PowerTransformer())  #PowerTransformer
                                ])

cat_transformer = Pipeline(steps=[
                                ('dataprep', data_transform(num_cols,cat_cols,is_cat=1)),
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='error')) #, drop = "if_binary"
                                ])

transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ori_cols),
        ('cat', cat_transformer, ori_cols)
    ])

knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=123)
rf = DecisionTreeRegressor(random_state=123)
eln = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=2020)
rg = Ridge(alpha=1.0, random_state=2020)
ls = Lasso(alpha=1.0, random_state=2020)
xgb = XGBRegressor(random_state=2020, booster='gbtree',n_estimators=20, tree_method='hist')
lgb = LGBMRegressor(objective='regression',random_states=2020, metric = 'rmse', num_leaves = 31, boosting_type='gbdt', max_depth=5, learning_rate=0.034)
catb = CatBoostRegressor(iterations=2,learning_rate=0.5,depth=3, silent=True)
# lgb_goss = lgb.LGBMRegressor(objective='regression',random_states=2020,  boosting_type = 'goss', metric = 'rmse', learning_rate=0.034, num_leaves = 31)
lr = LinearRegression()


stack = StackingCVRegressor(regressors=(knn, lgb, xgb, catb, dt, rf, eln, rg, ls), meta_regressor=lr, cv=3)

stack_pipeline = Pipeline(steps=[('transformer', transformer),
                      ('stack', stack)
                      ])

main_pipeline = TransformedTargetRegressor(stack_pipeline,
                                    transformer = PowerTransformer())

params = {  
#           'regressor__stack__lgbmregressor__boosting_type': ['gbdt','dart'],
#           'regressor__stack__lgbmregressor__max_depth': [3,5,7],
#           'regressor__stack__lgbmregressor__learning_rate': [0.05,0.5,1],
    
#           'regressor__stack__xgbregressor__n_estimators': [10,20,30],
#           'regressor__stack__xgbregressor__tree_method: ['exact','approx','hist'],
           
#           'regressor__stack__catboostregressor__n_estimators': [10,20,30],
#           'regressor__stack__catboostregressor__learning_rate': [0.05,0.5,1]
    
          }


model = GridSearchCV(estimator=main_pipeline, param_grid=params,  cv=3, n_jobs=-1, scoring='neg_mean_squared_error' ,refit=True)


train_X = train.drop(target,axis=1)
train_y = train[target]

model.fit(train_X, train_y)
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

def evaluate(model, train,val, label):
    """Mean absolute percentage error"""
    train_X = data_transform(num_cols,cat_cols,is_cat=0).transform(train)
    train_y = train[label]
    
    val_X = data_transform(num_cols,cat_cols,is_cat=0).transform(val)
    val_y = val[label]
    # Make predictions
    train_pred = model.predict(train_X)
    valid_pred = model.predict(val_X)
    
    # Get metrics
    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             train_y, val_y)
    
    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')
# model_lr_tyj = model #rmse val 200
#model_lr = model0 #rmse val 6.93
evaluate(model, train, val, target)
train_X = data_transform(num_cols,cat_cols,is_cat=0).transform(train)
train_y = train[target]
train_pred = model.predict(train_X)

plt.figure(figsize = (10, 6))
sns.kdeplot(train_y, label = 'Actual')
sns.kdeplot(train_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Train Data Fares");
val_X = data_transform(num_cols,cat_cols,is_cat=0).transform(val)
val_y = val[target]
val_pred = model.predict(val_X)

plt.figure(figsize = (10, 6))
sns.kdeplot(val_y, label = 'Actual')
sns.kdeplot(val_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of val Data Fares");
import joblib
# joblib.dump(model, f'nyk_taxi_lgbm.pkl')
# joblib.dump(model, f'nyk_taxi_lgbm_v2.pkl')
# joblib.dump(model, f'nyk_taxi_rf.pkl')
# joblib.dump(model, f'nyk_taxi_rf_yjt.pkl')
#joblib.dump(model, f'nyk_taxi_lgb_yjt_700k.pkl')
#joblib.dump(model, f'nyk_taxi_lgb_yjt_700k_tune1.pkl')
# joblib.dump(model, f'nyk_taxi_lgb_yjt_700k_tune2.pkl')
#joblib.dump(model, f'nyk_taxi_lgb_yjt_3mio_tune3.pkl')
#joblib.dump(model, f'nyk_taxi_lgb_yjt_3mio_goss.pkl')
model_gbdt = joblib.load(f'nyk_taxi_lgb_yjt_3mio_tune3.pkl')
model_goss = joblib.load(f'nyk_taxi_lgb_yjt_3mio_goss.pkl')
test = pd.read_csv('../input/test.csv', parse_dates = ['pickup_datetime'])
test.describe()
test_X = test.drop('key',axis=1)
# Use the model to make predictions
predicted_fare = model_gbdt.predict(test_X)*0.95553+ model_goss.predict(test_X)*0.00007
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_fare)
my_submission = pd.DataFrame({'key': test.key, 'fare_amount': predicted_fare})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_v10.csv', index=False)
my_submission.head()
dt    = '2019-06-20 12:26:21'
plong =  -73.844
plat  =  60.721
dlong =  123.842
dlat  =  60.712
pc    =  5
cek = pd.DataFrame({'pickup_datetime' : [dt], 'pickup_longitude' : [plong], 'pickup_latitude' : [plat], 'dropoff_longitude' : [dlong], 'dropoff_latitude' : [dlat], 'passenger_count' : [pc]})
cek = data_transform(num_cols,cat_cols,is_cat=0).transform(cek)
# model.predict(cek)
model_gbdt.predict(cek)*0.95553+ model_goss.predict(cek)*0.00007

class data_transform_v0():
    
    
    def __init__(self, num, cat):
        self.num = num
        self.cat = cat
        
    def fit(self, X):
        # do not do anything
        return self
    
    def transform(self, X, y = None):
        num_cols = self.num
        cat_cols = self.cat
        df = X
        df['passenger_count'] = np.where(df['passenger_count']<1,1,df['passenger_count'])
        df['passenger_count'] = np.where(df['passenger_count']>6,5,df['passenger_count'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']<40, 40,df['pickup_latitude'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']>42,42,df['pickup_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']<40, 40,df['dropoff_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']>42, 42,df['dropoff_latitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']<-75, -75,df['pickup_longitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']>-72, -72,df['pickup_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']<-75, -75,df['dropoff_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']>-72, -72,df['dropoff_longitude'])

        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        # Absolute difference in latitude and longitude
        df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
        df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()

        df['manhattan'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 1)

        df['euclidean'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 2)

        df['haversine'] =  haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                 df['dropoff_longitude'], df['dropoff_latitude']) 

        for i in place['loc'].tolist():
            for j in ['pickup','dropoff']:
                df[str(j)+'_distance_to'+str(i)] = distance_to_place(df,i,str(j)+'_longitude',str(j)+'_latitude')

        df['direction'] = calculate_direction(df)

        df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                                 time = True, start_ref = df['pickup_datetime'].min())

        df = df[num_cols+cat_cols]
        df[cat_cols] = df[cat_cols].astype(str)
            
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
num_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy = "mean")),
                                ('scaler', PowerTransformer())  #PowerTransformer
                                ])

cat_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='error')) #, drop = "if_binary"
                                ])
transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
#        ('cat', cat_transformer, cat_cols)
    ])

# main_pipeline = Pipeline(steps=[('transformer', transformer),
#                       ('lgbm', lgb.LGBMRegressor(objective='regression',random_states=2020))])

# param_grid = {
# #                 'lgbm__num_leaves': [30, 60, 90],
# #                 'lgbm__max_depth': [3, 5, 7],
# #                 'lgbm__n_estimators': [200, 400, 500],
#                 'lgbm__boosting' : ['goss'] #,'dart','gbdt'
#             }


# model = GridSearchCV(main_pipeline, param_grid, n_jobs=-1,scoring = 'neg_mean_absolute_error', cv=3)

model0 = Pipeline(steps=[('transformer', transformer),
                      ('lr', LinearRegression())])

model0_log = Pipeline(steps=[('transformer', transformer),
                      ('lr', LinearRegression())])

# model = TransformedTargetRegressor(model0,
#                                   func=np.log, 
#                                   inverse_func=np.exp)


train = train[train['fare_amount'].between(left = 2.5, right = 100)]
train = train.loc[(train['passenger_count'] > 0)&(train['passenger_count'] < 6)]
train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]

train_X = data_transform_v0(num_cols,cat_cols).transform(train)
train_y = train[target]

model0_log.fit(train_X, np.log(train_y))
model0.fit(train_X, train_y)

train_X = data_transform_v0(num_cols,cat_cols).transform(train)
train_y = train[target]
train_pred = model0_log.predict(train_X)
train_pred = np.exp(train_pred)

plt.figure(figsize = (10, 6))
sns.kdeplot(train_y, label = 'Actual')
sns.kdeplot(train_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Train Data Fares");
train_X = data_transform_v0(num_cols,cat_cols).transform(train)
train_y = train[target]
train_pred = model0.predict(train_X)

plt.figure(figsize = (10, 6))
sns.kdeplot(train_y, label = 'Actual')
sns.kdeplot(train_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Train Data Fares");
np.log(0)
for i in num_cols:
    print(i)
    sns.distplot(train[i])
    sns.distplot(np.log(train[i]))
    plt.show()
num_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy = "mean")),
                                ('scaler', PowerTransformer())  #PowerTransformer
                                ])

cat_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='error')) #, drop = "if_binary"
                                ])
transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
#        ('cat', cat_transformer, cat_cols)
    ])

# main_pipeline = Pipeline(steps=[('transformer', transformer),
#                       ('lgbm', lgb.LGBMRegressor(objective='regression',random_states=2020))])

# param_grid = {
# #                 'lgbm__num_leaves': [30, 60, 90],
# #                 'lgbm__max_depth': [3, 5, 7],
# #                 'lgbm__n_estimators': [200, 400, 500],
#                 'lgbm__boosting' : ['goss'] #,'dart','gbdt'
#             }


# model = GridSearchCV(main_pipeline, param_grid, n_jobs=-1,scoring = 'neg_mean_absolute_error', cv=3)

model0 = Pipeline(steps=[('transformer', transformer),
                      ('lr', LinearRegression())])

model = TransformedTargetRegressor(model0,
                                   transformer = PowerTransformer())


train = train[train['fare_amount'].between(left = 2.5, right = 100)]
train = train.loc[train['passenger_count'] < 6]
train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]

train_X = data_transform_v0(num_cols,cat_cols).transform(train)
train_y = np.array(train[target]).reshape(-1,1)


# model0.fit(train_X, np.log(train_y))
model.fit(train_X, train_y)

train['result_tyj_pipe'] =model.predict(train_X)
train[train['result_tyj_pipe']>100]
cek1 = train #[train['result_tyj_pipe']<=100]
plt.figure(figsize = (10, 6))
sns.kdeplot(cek1['fare_amount'], label = 'Actual')
sns.kdeplot(cek1['result_tyj_pipe'], label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Train Data Fares");
train1 = train.copy()
train1['fare_amount']
pt = PowerTransformer()
pt.fit(train1[['fare_amount']])

pt.transform(train1[['fare_amount']])
pt.inverse_transform(pt.transform(train1[['fare_amount']]))
pt.get_params(deep=True)
num_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy = "mean")),
                                ('scaler', PowerTransformer())  #PowerTransformer
                                ])

cat_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='error')) #, drop = "if_binary"
                                ])
transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
#        ('cat', cat_transformer, cat_cols)
    ])

# main_pipeline = Pipeline(steps=[('transformer', transformer),
#                       ('lgbm', lgb.LGBMRegressor(objective='regression',random_states=2020))])

# param_grid = {
# #                 'lgbm__num_leaves': [30, 60, 90],
# #                 'lgbm__max_depth': [3, 5, 7],
# #                 'lgbm__n_estimators': [200, 400, 500],
#                 'lgbm__boosting' : ['goss'] #,'dart','gbdt'
#             }


# model = GridSearchCV(main_pipeline, param_grid, n_jobs=-1,scoring = 'neg_mean_absolute_error', cv=3)

model = Pipeline(steps=[('transformer', transformer),
                      ('lr', LinearRegression())])


train = train[train['fare_amount'].between(left = 2.5, right = 100)]
train = train.loc[train['passenger_count'] < 6]
train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]

train_X = data_transform_v0(num_cols,cat_cols).transform(train)
train_y = train[[target]]

pt = PowerTransformer()
pt.fit(train_y)
train_yt = pt.transform(train_y)

# model0.fit(train_X, np.log(train_y))
model.fit(train_X, train_yt)

cek= model.predict(train_X)
pt.inverse_transform(cek).max()
(pt.inverse_transform(cek)>100).sum()
train['result_tyj'] = pt.inverse_transform(cek)
train[train['result_tyj'] >100]
cek1 = train#[train['result_tyj'] <=100]
sns.distplot(cek1['result_tyj'],color='r')
sns.distplot(cek1['fare_amount'],color='b')
