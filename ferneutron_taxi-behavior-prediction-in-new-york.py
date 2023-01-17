# Code created by: Fernando Lopez-Velasco

import pandas as pd
import datetime as dt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline
""" Uncomment this sentences to download the file """
#import os
#os.system('wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv')
data = pd.read_csv('../input/green-taxis/green_tripdata_2015-09.csv', parse_dates=True)
print(data.shape)
data.head()
data = data.drop(['VendorID','Store_and_fwd_flag',
                  'RateCodeID','Extra','MTA_tax',
                  'improvement_surcharge','Tip_amount',
                 'Tolls_amount','Trip_type ','Ehail_fee'], axis='columns')
data.head()
data.describe()
k = 0
for i in data.Fare_amount:
    if i < 0:
        k += 1
print("Number of fake fares: ", k)
data = data[data.Fare_amount>=0]
data = data[data.Total_amount>=0]
data.isnull().sum()
data['lon_change'] = abs(data.Dropoff_longitude - data.Pickup_longitude)
data['lat_change'] = abs(data.Dropoff_latitude - data.Pickup_latitude)
data.info()
data = data.drop(['Lpep_dropoff_datetime'], axis=1)
data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'])
data = data.set_index(['lpep_pickup_datetime'])
plt.style.use('fivethirtyeight')
data[data.Total_amount<70].Total_amount.hist(bins=70, figsize=(10,3))
plt.xlabel('Total fare $USD')
plt.title("Histogram of total fare's distribution");
data[data.Trip_distance<10].Trip_distance.hist(bins=100, figsize=(10,3))
plt.xlabel('Total Distance')
plt.title("Histogram of total trip distribution");
data.shape
print(data.Trip_distance.mean())
print(data.Trip_distance.std())
plot_sns = sns.countplot(data.Passenger_count, label='Passengers')
data.Passenger_count.mean()
plot_sns = sns.countplot(data.Payment_type, label='Number of payments')
type1,type2,type3,type4,type5 = data.Payment_type.value_counts()
print("Payment with type 1: ", type1)
print("Payment with type 2: ", type2)
print("Payment with type 3: ", type3)
print("Payment with type 4: ", type4)
print("Payment with type 5: ", type5)
# Create a new dataframe to be used in time series plotting
data_plot = data[['Trip_distance','Passenger_count', 'Total_amount']]

# Get different indexes: day, hour, minute.
index_day = data_plot.index.day
index_hour = data_plot.index.hour
index_minute = data_plot.index.minute

# Get the column's name from data_plot
distance, NumPassenger, Total = data_plot.columns
# Defining a function to plot time series
def plotting(item, index):
    
    plt.style.use('fivethirtyeight')
    if item=='Trip_distance':
        ylabel = 'Distance'
    elif item=='Passenger_count':
        ylabel = 'Passengers'
    elif item=='Total_amount':
        ylabel = 'Fare $USD'
        
    to_plot = data_plot[item].groupby(index).mean()
    hline = to_plot.max()
    vline = to_plot.idxmax()
    ax = to_plot.plot(linewidth=0.8,figsize=(15,3))
    ax.axvline(vline, color='red', linewidth=0.9,linestyle='--')
    ax.axhline(hline, color='green', linewidth=0.9, linestyle='--')
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    print("Max in time: ", vline)
    print("Max in distance: ",hline)
    plt.show()
# Grouping the distance by hour
trip = data_plot[distance].groupby(index_hour).mean()
print("Mean: ", trip.mean())
print("Median: ",trip.median())
ax = trip.plot(figsize=(6,4), linewidth=0.9)
ax.set_xlabel('Hour')
ax.axhline(trip.median(), color='red', linewidth=0.6, linestyle='--')
ax.axhline(trip.mean(), color='purple', linewidth=0.6, linestyle='--')
ax.axhspan(trip.mean(),trip.median(), color='blue', alpha=0.1)
ax.set_ylabel('Avergae')
plotting(distance, index_day)
plotting(distance, index_hour)
plotting(distance, index_minute)
plotting(NumPassenger, index_day)
plotting(NumPassenger, index_hour)
plotting(NumPassenger, index_minute)
plotting(Total, index_day)
plotting(Total, index_hour)
plotting(Total, index_minute)
corr_ = data_plot.corr(method='pearson')
sns.heatmap(corr_,annot=True,linewidths=0.4,annot_kws={"size": 10})
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
corr_ = data_plot.corr(method='pearson')
fig = sns.clustermap(corr_, row_cluster=True,col_cluster=True,figsize=(5, 5))
plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
# Ideas and pieces of code for this functiones were taken from: Albert van Breemen

jfk = (-73.7822222222, 40.6441666667) #JFK Airport
nyc = (-74.0063889, 40.7141667) # NYC Airport
ewr = (-74.175, 40.69) # Newark Liberty International Airport
lgr = (-73.87, 40.77) # LaGuardia Airport

# Function to calculate distances given coordenates
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

# Function to plot relations between distances and total fare amount
def plot_location_fare(loc, name, range=1):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    idx = (distance(data.Pickup_latitude, data.Pickup_longitude, loc[1], loc[0]) < range)
    data[idx].Fare_amount.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel('Total Fare', fontsize=8)
    axs[0].set_title('Pickup location \n{}'.format(name), fontsize=8)

    idx = (distance(data.Dropoff_latitude, data.Dropoff_longitude, loc[1], loc[0]) < range)
    data[idx].Fare_amount.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel('Total Fare', fontsize=8)
    axs[1].set_title('Dropoff location \n{}'.format(name), fontsize=8);
plot_location_fare(jfk, 'JFK Airport')
plot_location_fare(ewr, 'Newark Airport')
plot_location_fare(lgr, 'LaGuardia Airport')
# Set tip as 13% from total amount
tip = 0.13
data['tip'] = data['Total_amount'] + data['Total_amount'] * tip
data.head()
# Define my frame for this excercise
data_model = data[['Trip_distance','Fare_amount','Total_amount','tip']]
# Split target and corpus data
Y = data_model.tip
X = data_model.drop(['tip'], axis=1)
X.head()
# Load libraries and stuff

import xgboost as xgb # xgboost regressor
from sklearn import preprocessing # to use min_max_scaler
from sklearn.model_selection import train_test_split # split data
from sklearn.metrics import mean_absolute_error, mean_squared_error # metrics
# Initialize the object and fit
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns = X.columns)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
# to plot performance
def plot_performance(plot_name, loss_mae, loss_mse):
    steps = np.arange(10, 100, 20)
    plt.title(plot_name)
    plt.plot(steps, loss_mae, linewidth=0.9, label="MAE")
    plt.plot(steps, loss_mse, linewidth=0.9, label="MSE")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Number of estimators")
    plt.show()

# to calculate metrics
def def_metrics(Ypred):
    mae = mean_absolute_error(Ytest, Ypred)
    mse = mean_squared_error(Ytest, Ypred)

    return mae, mse

# to generate the model
def def_xgboost(estimators):
    xgb_ = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.1, max_depth=3, n_estimators=estimators)
    xgb_.fit(Xtrain,Ytrain)
    Ypred = xgb_.predict(Xtest)
    
    return def_metrics(Ypred)
loss_mae, loss_mse = [], []
plot_name="Tip Prediction"
for est in range(10,100,20):
    print("Number of estimators: %d" % est)
    mae, mse = def_xgboost(estimators = est)
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)
# Ideas and pieces of code for this functiones were taken from: Albert van Breemen

# Defining the box to plot
def select_within_boundingbox(data, BB):
    return (data.Pickup_longitude >= BB[0]) & (data.Pickup_longitude <= BB[1]) & \
           (data.Pickup_latitude >= BB[2]) & (data.Pickup_latitude <= BB[3]) & \
           (data.Dropoff_longitude >= BB[0]) & (data.Dropoff_longitude <= BB[1]) & \
           (data.Dropoff_latitude >= BB[2]) & (data.Dropoff_latitude <= BB[3])
            
# Loading the image of NYC
BB = (-74.5, -72.8, 40.5, 41.8)
nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')

# Loading an image with zoom in NYC
BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')
# Defining frame for visualization
data_box = data[select_within_boundingbox(data, BB)]
# Ideas and pieces of code for this functiones were taken from: Albert van Breemen
# Function to plot dots on map
def plot_on_map(data_box, BB, nyc_map, s=10, alpha=0.2):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].scatter(data_box.Pickup_longitude, data_box.Pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title('Pickup locations')
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(data_box.Dropoff_longitude, data_box.Dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].imshow(nyc_map, zorder=0, extent=BB)
# Plotting the map with dots
plot_on_map(data_box, BB, nyc_map, s=1, alpha=0.5)
# Ideas and pieces of code for this functiones were taken from: Albert van Breemen
# Setting the number of bins and fitting in pickup and dropoff
n_lon, n_lat = 200, 200
density_pickup, density_dropoff = np.zeros((n_lat, n_lon)), np.zeros((n_lat, n_lon))

# Calculating the number of datapoint in the grid
bins_lon = np.zeros(n_lon+1) # bin
bins_lat = np.zeros(n_lat+1) # bin

delta_lon = (BB[1]-BB[0]) / n_lon # bin longutide width
delta_lat = (BB[3]-BB[2]) / n_lat # bin latitude height

bin_width_miles = distance(BB[2], BB[1], BB[2], BB[0]) / n_lon # bin width in miles
bin_height_miles = distance(BB[3], BB[0], BB[2], BB[0]) / n_lat # bin height in miles

for i in range(n_lon+1):
    bins_lon[i] = BB[0] + i * delta_lon
for j in range(n_lat+1):
    bins_lat[j] = BB[2] + j * delta_lat
    
# Digitalization by pickup, dropoff given bins on longitude and latitude
inds_pickup_lon = np.digitize(data_box.Pickup_longitude, bins_lon)
inds_pickup_lat = np.digitize(data_box.Pickup_latitude, bins_lat)
inds_dropoff_lon = np.digitize(data_box.Dropoff_longitude, bins_lon)
inds_dropoff_lat = np.digitize(data_box.Dropoff_latitude, bins_lat)

# Assign a point by grid bin
dxdy = bin_width_miles * bin_height_miles
for i in range(n_lon):
    for j in range(n_lat):
        density_pickup[j, i] = np.sum((inds_pickup_lon==i+1) & (inds_pickup_lat==(n_lat-j))) / dxdy
        density_dropoff[j, i] = np.sum((inds_dropoff_lon==i+1) & (inds_dropoff_lat==(n_lat-j))) / dxdy
# Ideas and pieces of code for this functiones were taken from: Albert van Breemen
# Plot the density arrays
fig, axs = plt.subplots(2, 1, figsize=(18, 24))
axs[0].imshow(nyc_map, zorder=0, extent=BB);
im = axs[0].imshow(np.log1p(density_pickup), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
axs[0].set_title('Pickup density [datapoints per sq mile]')
cbar = fig.colorbar(im, ax=axs[0])
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)

axs[1].imshow(nyc_map, zorder=0, extent=BB);
im = axs[1].imshow(np.log1p(density_dropoff), zorder=1, extent=BB, alpha=0.6, cmap='plasma')
axs[1].set_title('Dropoff density [datapoints per sq mile]')
cbar = fig.colorbar(im, ax=axs[1])
cbar.set_label('log(1 + #datapoints per sq mile)', rotation=270)
