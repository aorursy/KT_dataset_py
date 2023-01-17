# packages to load in 
import numpy as np                                   # linear algebra
import pandas as pd                                  # data processing, CSV file I/O (e.g. pd.read_csv)
import math                                          # from math import cos, asin, sqrt
import warnings                                      #ignore warnings
warnings.filterwarnings('ignore')                    # supress the warning messages
import dask                                          # distributed parallel processing
import dask.dataframe as dd                          # data processing, CSV file I/O (e.g. pd.read_csv), dask
from dask.distributed import Client, progress        # task distribution
client = Client()
import seaborn as sns, matplotlib.pyplot as plt      # visualizations
% matplotlib inline

import folium                                        # map visualizations
from folium.plugins import HeatMap                  # map visualizations - heatmap 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# input file path 
data_train_file_path =   "../input/train.csv"
data_test_file_path = "../input/test.csv"
client
%%time
db_rows_count = 55423857
with open(data_train_file_path) as f:
    db_rows_count = len(f.readlines())   
print("no. of rows in the training data : {0}\n".format(db_rows_count))
# training data - Set columns to most suitable type to optimize for memory usage and speed-up the loading
train_types = {'fare_amount'      : 'float32',
               'pickup_datetime'  : 'str', 
               'pickup_longitude' : 'float32',
               'pickup_latitude'  : 'float32',
               'dropoff_longitude': 'float32',
               'dropoff_latitude' : 'float32'}

# test-data - Set columns to most suitable type to optimize for memory usage and speed-up the loading
test_types = { 'pickup_datetime'  : 'str',
                'key'             : 'str',
               'pickup_longitude' : 'float32',
               'pickup_latitude'  : 'float32',
               'dropoff_longitude': 'float32',
               'dropoff_latitude' : 'float32'}


# select the columns (names) that you truly need for analysis - training data
train_cols = list(train_types.keys())    

# select the columns (names) that you truly need for analysis - test data
test_cols = list(test_types.keys())  

# NY city - defining the bounding box
BB = (-74.5, -72.8, 40.5, 41.8)        

# set the amount of data to load from db
frac = 0.00050                   # set the amount of data to load from db
    
# select within the bounding box
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])


def load_data(data_file_path, train_data='Y'):
    
    # training data load and filter inputs    
    if (train_data=='Y'):
        df = dd.read_csv(data_file_path,usecols=train_cols, dtype=train_types)  # data load, dask
        
        column_names = ["fare_amount",
                        "pickup_longitude",
                        "pickup_latitude", 
                        "dropoff_longitude",
                        "dropoff_latitude"]                          # selecting the columns to check for empty values
        df = df.sample(frac=0.04)                                    # percentage of rows to load.  loading 2 million rows
        df = df.dropna(how="any", subset = train_cols)               # remove rows with null values
        df = df[(df[column_names] != 0).all(axis=1)]                 # remove the latitude and longitude rows with zeros
        df = df.loc[(df.fare_amount > 0) & (df.fare_amount < 100) & 
            ~(((df.pickup_longitude - df.dropoff_longitude) == 0) & 
             ((df.pickup_latitude - df.dropoff_latitude) == 0))]     # remove the rows where fare amounts less than or greater than zero or with same coordinates
        df = df[select_within_boundingbox(df, BB)]                   #remove the coordinates not within the newyork city
    
    if (train_data == 'N'):
        df = dd.read_csv(data_file_path,usecols=test_cols, dtype=test_types)  # data load, dask
    
    df = dd.concat([
        df,dd.to_datetime(df['pickup_datetime']).apply(
        lambda x: pd.Series([x.year, x.month, x.day, x.weekday(), x.hour, x.minute],
        index=['pickup_year', 'pickup_month', 'pickup_dd' ,'pickup_weekday', 'pickup_hour', 'pickup_minute']))], axis=1)   # extract year, month, weekday and hour from pickup datetime  

    df = client.persist(df) 
    return df

#call the subroutine to load the data 
df = load_data(data_train_file_path, 'Y')
progress(df)
# number of rows from db and after applying the filters
after = len(df)
print('# of rows in training data \n\t actual : {0}  \n\t after applying filters : {1}  \n\t dropped rows: {2} '.format(db_rows_count, after, db_rows_count-after))   # before and after filter rows count 
#top 10 rows
df.head(10)
%%time
print(f'# of rows processing : {len(df)}')
print("\033[4m\nColumn Name\tisnull_counts\tdata_types\033[0m")
for columns in df.columns:
    print(f'{columns.ljust(17)}\t{(df[columns].isnull().map_partitions(sum).compute().sum()):>5}\t{(df[columns].dtype)}')
print('\t')
# distance calculation in Kilometeres
from math import cos, asin, sqrt

def distance_haversine(lon1, lat1, lon2, lat2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) * 0.62137 #2*R*asin...
def distance_rows(p_lon, p_lat, d_lon, d_lat):
    nyc_coord = (40.7141667, -74.0063889,)      # ny city center coordinates
    jfk_coord = (40.639722, -73.778889)         #John F. Kennedy International Airport coordinates
    ewr_coord = (40.6925, -74.168611)           #Newark Liberty International Airport coordinates
    lga_coord = (40.77725, -73.872611)          #LaGuardia Airport coordinates

    distance_between_pickup_dropoff = distance_haversine(p_lon, p_lat, d_lon, d_lat)                    # distance between pickup and dropff
    distance_between_pickup_jfk     = distance_haversine(p_lon, p_lat, jfk_coord[1], jfk_coord[0])      # distance between pickup and jfk airport
    distance_between_dropoff_jfk    = distance_haversine(jfk_coord[1], jfk_coord[0], d_lon, d_lat)      # distance between dropoff and jfk airport
    distance_between_pickup_ewr     = distance_haversine(p_lon, p_lat, ewr_coord[1], ewr_coord[0])      # distance between pickup and ewr airport
    distance_between_dropoff_ewr    = distance_haversine(ewr_coord[1], ewr_coord[0], d_lon, d_lat)      # distance between dropoff and ewr airport
    distance_between_pickup_lga     = distance_haversine(p_lon, p_lat, lga_coord[1], lga_coord[0])      # distance between pickup and lga airport
    distance_between_dropoff_lga    = distance_haversine(lga_coord[1], lga_coord[0], d_lon, d_lat)      # distance between dropoff and lga airport
    distance_between_citycenter_pickup = distance_haversine(nyc_coord[0], nyc_coord[1],p_lon, p_lat)    # distance between pickup and city center
    longitude_diff                     = p_lon - d_lon
    latitude_diff                      = p_lat - d_lat
    
    return [distance_between_pickup_dropoff,
            distance_between_pickup_jfk,
            distance_between_dropoff_jfk, 
            distance_between_pickup_ewr, 
            distance_between_dropoff_ewr, 
            distance_between_pickup_lga, 
            distance_between_dropoff_lga,
            distance_between_citycenter_pickup,
            longitude_diff,
            latitude_diff]

def calculate_coordinates_distance(df, train_data='Y'):
    # distance columns to be added to the data frame
    column_names  = ['distance_between_pickup_dropoff', 
                     'distance_between_pickup_jfk', 
                     'distance_between_dropoff_jfk', 
                     'distance_between_pickup_ewr', 
                     'distance_between_dropoff_ewr', 
                     'distance_between_pickup_lga', 
                     'distance_between_dropoff_lga',
                     'distance_between_citycenter_pickup',
                     'longitude_diff',
                     'latitude_diff']

    # pandas dataframes processing - utilizing dask
    df = dd.concat([df,df[["pickup_longitude","pickup_latitude", "dropoff_longitude","dropoff_latitude"]].apply(lambda x: pd.Series(distance_rows(*x),index=column_names), axis=1)], axis=1)

    # calculate fare per mile
    if (train_data == 'Y'):
        # remove data points less than .05 miles
        df = df.loc[df.distance_between_pickup_dropoff>0.05]
        df['fare_per_mile'] = df.fare_amount/df.distance_between_pickup_dropoff 

    #reset the index
    df = df.reset_index(drop=True)  
    df = client.persist(df)
    return df

# calculate the distance between the coordinates
df = calculate_coordinates_distance(df, train_data='Y')
progress(df)
df.compute().info()
#create a map
this_map = folium.Map(location=[40.741895, -73.989308],
                      zoom_start=11
)

def plotDot(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.CircleMarker(location=[point.pickup_latitude, point.pickup_longitude],
                        radius=2,color='#3186cc', fill=True,fill_color='#3186cc',
                       weight=0).add_to(this_map)

df.compute().head(5000).apply(plotDot, axis = 1)

#Set the zoom to the maximum possible 
#this_map.fit_bounds(this_map.get_bounds())
    
this_map  
#create a map
this_map = folium.Map(location=[40.741895, -73.989308])

# List comprehension to make out list of lists
heat_data = [[row['pickup_latitude'],row['pickup_longitude']] for index, row in df.compute().iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(this_map)

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())
    
this_map  
@dask.delayed
def round_decimals(x, x_decimals=2):
    return x.round(x_decimals)

@dask.delayed
def math_sqrt(x):
    return math.sqrt(x)
fare_amount_mean = df["fare_amount"].mean()
fare_amount_standard_deviation = math_sqrt(((df["fare_amount"] - fare_amount_mean) ** 2).mean())

print("average fair amount (mean) : ${0:.2f}".format(fare_amount_mean.compute()))
print("fare amount standard deviation : ${0:.2f}\n".format(fare_amount_standard_deviation.compute()))
# plot histogram of fare
plt.figure(figsize=(25,10))
sns.set(color_codes=True)
ax = sns.distplot(df.fare_amount, bins=15, kde=False)
plt.xlabel('fare $USD', fontsize=20)
plt.ylabel('frequency', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
    
plt.title('fare amount Histogram', fontsize=25)
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(20,12))
# Plot the responses for different events and regions
sns.lineplot(x="pickup_hour", y="fare_per_mile",
            hue="pickup_year", style="pickup_year",  dashes=False, 
             data=df.compute())
plt.show()
df_day              = df.loc[(df.pickup_hour >=6) & (df.pickup_hour <16)]
df_peak_hours       = df.loc[((df.pickup_hour >=16) & (df.pickup_hour <20))]
df_night            = df.loc[~((df.pickup_hour >=6) & (df.pickup_hour <20))]
df_between_airports = df.loc[(((df.distance_between_pickup_jfk < 2)  | (df.distance_between_pickup_ewr < 2)  | (df.distance_between_pickup_lga < 2))  &
                              ((df.distance_between_dropoff_jfk < 2) | (df.distance_between_dropoff_lga < 2) | (df.distance_between_dropoff_ewr < 2))                      )] 
df_airport_pickup   = df.loc[((df.distance_between_pickup_jfk < 2)  | (df.distance_between_pickup_ewr < 2)  | (df.distance_between_pickup_lga < 2))]   
df_airport_dropoff  = df.loc[((df.distance_between_dropoff_jfk < 2) | (df.distance_between_dropoff_lga < 2) | (df.distance_between_dropoff_ewr < 2))] 

#remove the coordinates not within the newyork city
BB_manhattan = (-74.025, 40.7, -73.925, 40.8)
df_jfk_manhattan = df[(select_within_boundingbox(df, BB_manhattan) &
                      ((df.distance_between_pickup_jfk < 2) | (df.distance_between_dropoff_jfk < 2)))]

#reset the index
df = df.reset_index(drop=True)  
%%time

fare_amount_per_mile                  = df.fare_per_mile.mean().compute().round(2)

fare_amount_per_mile_day              = df_day.fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_peak_hours       = df_peak_hours.fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_night            = df_night.fare_per_mile.mean().compute().round(2)

fare_amount_per_mile_between_airports = df_between_airports.fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_airport_pickup   = df_airport_pickup.fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_airport_dropoff  = df_airport_dropoff.fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_jfk_manhattan    = df_jfk_manhattan.fare_per_mile.mean().compute().round(2)

fare_amount_per_mile_weekday          = df.loc[df.pickup_weekday<=4].fare_per_mile.mean().compute().round(2)
fare_amount_per_mile_weekend          = df.loc[df.pickup_weekday>=5].fare_per_mile.mean().compute().round(2)

avg_data = pd.DataFrame({'fare':[
                    fare_amount_per_mile_between_airports,  
                    fare_amount_per_mile_jfk_manhattan,
                    fare_amount_per_mile_airport_pickup,
                    fare_amount_per_mile_airport_dropoff,
                    
                    fare_amount_per_mile_weekend,
                    fare_amount_per_mile_weekday,
                    
                    fare_amount_per_mile_peak_hours,
                    fare_amount_per_mile_night, 
                    fare_amount_per_mile_day,
                    fare_amount_per_mile
]}, index = [ 
             'between_airports', 
             'jfk_manhattan',      
             'airport_pickup', 
             'airport_dropoff', 
             'week end',  
             'week day', 
             'peak hours(4-8pm)',
             'night_ride',
             'day_ride',
             'all_day']
).dropna()
# average fare
sns.set_style("white")
plt.figure(figsize=(20,8))
plt.barh(avg_data.index, avg_data.fare, height = .4, align='center',  color="b")
plt.title("fare per mile - trip average", fontsize=20)
plt.xlabel('fare $ USD', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for i, v in enumerate(avg_data.fare):
    plt.text(v,i-.1, '$' + str(v), fontsize=12)
fare_per_mile_yr                  = df.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_day_yr              = df_day.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_peak_hours_yr       = df_peak_hours.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_night_yr            = df_night.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_weekday_yr          = df.loc[df.pickup_weekday<=4].groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_weekend_yr          = df.loc[df.pickup_weekday>=5].groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_airport_pickup_yr   = df_airport_pickup.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_airport_dropoff_yr  = df_airport_dropoff.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_between_airports_yr = df_between_airports.groupby('pickup_year')['fare_per_mile'].mean().compute() 
fare_per_mile_jfk_manhattan_yr    = df_jfk_manhattan.groupby('pickup_year')['fare_per_mile'].mean().compute() 

fare_mile = pd.concat([
            pd.DataFrame({'year':fare_per_mile_yr.index, 'avg_fare':fare_per_mile_yr.values, 'type':'overall'}),
            pd.DataFrame({'year':fare_per_mile_day_yr.index, 'avg_fare':fare_per_mile_day_yr.values, 'type':'day_time'}),
            pd.DataFrame({'year':fare_per_mile_peak_hours_yr.index, 'avg_fare':fare_per_mile_peak_hours_yr.values, 'type':'peak_hours'}),
            pd.DataFrame({'year':fare_per_mile_night_yr.index, 'avg_fare':fare_per_mile_night_yr.values, 'type':'night_time'}),
            pd.DataFrame({'year':fare_per_mile_weekday_yr.index, 'avg_fare':fare_per_mile_weekday_yr.values, 'type':'weekend'}),
            pd.DataFrame({'year':fare_per_mile_weekend_yr.index, 'avg_fare':fare_per_mile_weekend_yr.values, 'type':'weekday'}),
            pd.DataFrame({'year':fare_per_mile_airport_pickup_yr.index, 'avg_fare':fare_per_mile_airport_pickup_yr.values, 'type':'airport_pickup'}),
            pd.DataFrame({'year':fare_per_mile_airport_dropoff_yr.index, 'avg_fare':fare_per_mile_airport_dropoff_yr.values, 'type':'airport_dropoff'}),
            pd.DataFrame({'year':fare_per_mile_jfk_manhattan_yr.index, 'avg_fare':fare_per_mile_jfk_manhattan_yr.values, 'type':'jfk_manhattan'}),
            pd.DataFrame({'year':fare_per_mile_between_airports_yr.index, 'avg_fare':fare_per_mile_between_airports_yr.values, 'type':'between_airports'}),
            ]).reset_index(drop = True)   

plt.figure(figsize=(20,12)) 
ax = sns.barplot(x="type", y="avg_fare", hue="year", data=fare_mile, palette="Blues")
plt.title("fare per mile - trip average", fontsize=16)
plt.ylabel('fare $ USD', fontsize=11)
plt.xlabel('')
plt.xticks(fontsize=11, rotation =90)
plt.yticks(fontsize=11)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height-1.5,
            '${:1.2f}'.format(height),
            ha="center", rotation=90) 
df.compute().info()
df.head(10)
%%time
import xgboost as xgb
import dask_xgboost as dxgb
from sklearn.metrics import mean_squared_error

X =  df.drop(['fare_amount', 'fare_per_mile', 'pickup_datetime'], axis=1)
y =  df.fare_amount   

X_train, X_test = X.random_split([0.7, 0.3], random_state=0)
y_train, y_test = y.random_split([0.7, 0.3], random_state=0)

def dxgb_evaluate() :
    params = {'eval_metric'        : 'rmse' 
              ,'num_boost_round'   : 100
              ,'max_depth'         : 7
              ,'seed'              : 0
              ,'subsample'         : 0.8 
              ,'silent'            : True 
              ,'gamma'             : 1
              ,'colsample_bytree'  : 0.9
              ,'nfold'             : 3 
              ,'boosting_type'     : 'gbdt'
              , 'seed' : 0
         }

    bst = dxgb.train(client, params, X_train, y_train)
    del(params)
    return bst

# train the model
bst = dxgb_evaluate()
# train split predictions
X_train_predictions = dxgb.predict(client, bst, X_train)

# train test split predictions
X_test_predictions = dxgb.predict(client, bst, X_test)
# Report testing and training RMSE
print("\033[1;37;40m\033[2;37:40mdata category \t\t\trmse-score\033[0m")
print('train test split \t\t\033[0;37;41m  {0:.2f}  \033[0m'.format(np.sqrt(mean_squared_error(y_test, X_test_predictions))))
print('train split \t\t\t\033[0;37;41m  {0:.2f}  \033[0m\n'.format(np.sqrt(mean_squared_error(y_train, X_train_predictions))))
fig, ax = plt.subplots(figsize=(12, 8))
ax = xgb.plot_importance(bst, ax=ax, height=0.8, max_num_features=20, color='b')
ax.grid("off", axis="y")
#for i in range(100):
#    print(s[i], y_pred[i].round(1))
%%time
#load the data 
df_test = load_data(data_test_file_path, train_data='N')
df_test = calculate_coordinates_distance(df_test, train_data='N')

df_test_key = df_test.key
df_test     = df_test.drop(['key', 'pickup_datetime'], axis=1)
df_test.compute().info()
df_test.head(5)
# train split predictions
test_predictions = dxgb.predict(client, bst, df_test)
submission_predictions  = pd.DataFrame({'key': df_test_key.compute(), 'fare_amount': test_predictions.compute()})
submission_predictions.to_csv('submission.csv', index=False)
submission_predictions
print(os.listdir('.'))