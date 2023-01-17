# Let's start with standard libraries

# The seed setting is an attempt to get consistent results from Keras. 
from numpy.random import seed
seed(20)
from tensorflow import set_random_seed
set_random_seed(40)

import keras
from keras import backend as K

import tensorflow as tf
import os

#import glob
#import datetime 

print(os.listdir("../input"))
# We will set some configuration variables here
preTrainedModel = False
loadOnlyWeights = False
useBestModel = True

EPOCH = 32
#EPOCH = 40

BATCH_SIZE = 1024

VALIDATION_SIZE = 500000
#TRAINING_SIZE = 10000000 #6M
TRAINING_SIZE = 6000000 #6M
#TRAINING_SIZE = 60000

#INTEL_AI_DEV_CLOUD = True
INTEL_AI_DEV_CLOUD = False

# This code is added to enable the current notebook to work seamlessly on Intel AI DevCloud. 
# But it turns out that my MacBook is much better than Intel AI DevCloud resources.

if INTEL_AI_DEV_CLOUD:
    
    # Set environment variables to fine tune the nodes for better performance
    config = tf.ConfigProto(intra_op_parallelism_threads = 6, 
                            inter_op_parallelism_threads = 2, 
                            allow_soft_placement = True, 
                            device_count = {'CPU': 6 })

    session = tf.Session(config = config)

    K.set_session(session) 
    os.environ["OMP_NUM_THREADS"] = "6" 
    os.environ["KMP_BLOCKTIME"] = "30" 
    os.environ["KMP_SETTINGS"] = "1" 
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    
# end of if statement
# Define the necessary libraries required to define Keras Regression Netowrk
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# import our good friend numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

print(keras.__version__)

# Many thanks to Albert Van Breemen for inspiring the idea of fixed frame of reference

# JFK airport & NYC coordinates, see https://www.travelmath.com/airport/JFK
# Latitude: 40.6441666667 / Longitude: -73.7822222222
# Latitude: 40.7141667 / Longitude: -74.0063889

JFK = (-73.7822222222, 40.6441666667)
NYC = (-74.0063889, 40.7141667)

# We will use EWR and LGR only if required - too many features are also a problem
EWR = (-74.175, 40.6897222222)
LGA = (-73.8739659, 40.7769271)
# Many thanks to Will Koehrsen for this wonderful function
# x1 = pickup_longitude / x2 = dropoff_longitude / y1 = pickup_latitude / y2 = dropoff_latitude
# if p = 1, then this is the Manhattan distance and if p = 2 this is the Euclidean distance.

def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)        
# if p = 1, then this is the Manhattan distance and if p = 2 this is the Euclidean distance.
# Inspired by 3 dimensional representation of latitude and longitude - 
# please see the details below regarding converting latitude and longitude to three dimensional coordinates.
def minkowski_distance_3D(x1, x2, y1, y2, z1, z2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1) ** p) + (abs(z2 - z1) ** p)) ** (1 / p)
# MANY MANY THANKS TO Albert Van Breeman for this useful function!
# direction of a trip, from 180 to -180 degrees. Horizontal axes = 0 degrees.
def calculate_direction(d_lon, d_lat):
    
    result = np.zeros(len(d_lon))
    
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    
    return result

#df_train['direction'] = calculate_direction(df_train.delta_lon, df_train.delta_lat)
# Another function ispired by Albert Van Breeman's Analysis of NY Taxi Data
def time_type_of_the_day(hour):
    
    time_of_the_day = 2
    
    if (hour >= 4 and hour < 7):
        time_of_the_day = 1 # Early Morning
    elif (hour >= 7 and hour < 12):
        time_of_the_day = 2 # Morning
    elif (hour >= 12 and hour < 16):
        time_of_the_day = 3 # Noon
    elif (hour >= 16 and hour < 20):
        time_of_the_day = 4 # Evening
    elif (hour >= 20 and hour < 23):
        time_of_the_day = 5 # Night
    elif (hour >= 23 and hour < 4):
        time_of_the_day = 6 # Late Night
    else:
        time_of_the_day = 2 # Morning Default

    return time_of_the_day

# end of the function!
# The day of the week with Monday=0, Sunday=6
# 0.5 of additional surcharge between 8PM - 6AM.
# Peak hour weekday surcharge of $1 Monday-Friday between 4PM-8PM.
def peak_hours_of_the_day(hour, weekday):
    
    peak_hours = 0
    
    if (0 <= weekday <= 4) and (16 <= hour <= 20):
        peak_hours = 1
        
    return peak_hours

# end of the function!
# The day of the week with Monday=0, Sunday=6
# 0.5 of additional surcharge between 8PM - 6AM.
# Peak hour weekday surcharge of $1 Monday-Friday between 4PM-8PM.
def surcharge_hours_of_the_day(hour):
    
    surcharge_hours = 0
    
    if (hour >= 20 or hour <= 6):
        surcharge_hours = 1
        
    return surcharge_hours

# end of the function!
def add_travel_vector_features(df):
    
    df['delta_longitude'] = df.pickup_longitude - df.dropoff_longitude
    df['delta_latitude'] = df.pickup_latitude - df.dropoff_latitude

    # The credit for the next features goes to Jan van der Vegt @datascience.stackexchange.com
    # https://datascience.stackexchange.com/users/14904/jan-van-der-vegt
    
    # Lat long coordinates have a problem that they are 2 features that represent a three dimensional space. 
    # This means that the long coordinate goes all around, which means the two most extreme values are actually very close together. 
    # I've dealt with this problem a few times and what I do in this case is map them to x, y and z coordinates. 
    # This means close points in these 3 dimensions are also close in reality. 
    # Depending on the use case you can disregard the changes in height and map them to a perfect sphere. 
    # These features can then be standardized properly.

    # To clarify (summarised from the comments):

    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon), 
    # z = sin(lat) 
    
    # The mapping of pickup and dropoff to three dimensional space helped to improve my ranking by 80 places. It is very similar 
    # to two dimensional representation of cyclical values.
    
    df['pickup_x'] = np.cos(df.pickup_latitude) * np.cos(df.pickup_longitude)
    df['pickup_y'] = np.cos(df.pickup_latitude) * np.sin(df.pickup_longitude)
    df['pickup_z'] = np.sin(df.pickup_latitude)
    
    df['dropoff_x'] = np.cos(df.dropoff_latitude) * np.cos(df.dropoff_longitude)
    df['dropoff_y'] = np.cos(df.dropoff_latitude) * np.sin(df.dropoff_longitude)
    df['dropoff_z'] = np.sin(df.dropoff_latitude)

# end of function add_travel_vector_features

def parse_and_extract_date_time_components(df):
        
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
      
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['minute'] = df['pickup_datetime'].apply(lambda x: x.minute)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.dayofweek)
    df['weekofyear'] = df['pickup_datetime'].apply(lambda x: x.weekofyear)
    df['dayofyear'] = df['pickup_datetime'].apply(lambda x: x.dayofyear)
    df['quarterofyear'] = df['pickup_datetime'].apply(lambda x: x.quarter)
    df['days_in_month'] = df['pickup_datetime'].apply(lambda x: x.days_in_month)
    df['linear_time'] = df['pickup_datetime'].apply(lambda x: x.value // 10 ** 9)    
    df['time_type_of_the_day'] = df['pickup_datetime'].apply (lambda x: time_type_of_the_day(x.hour)) 
    df['peak_hours'] = df['pickup_datetime'].apply(lambda x: peak_hours_of_the_day(x.hour, x.dayofweek))
    df['surcharge'] = df['pickup_datetime'].apply(lambda x: surcharge_hours_of_the_day(x.hour))
    
    # Process cyclical values using sin and cosine transformations
    # These are better than one-hot encoding and helped me a lot to keep features compact.
    
    df['hour_sin'] = np.sin(df.hour * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour * (2. * np.pi / 24))
 
    df['minute_sin'] = np.sin(df.minute * (2. * np.pi / 60))
    df['minute_cos'] = np.cos(df.minute * (2. * np.pi / 60))
    
    df['time_type_of_day_sin'] = np.sin(df.time_type_of_the_day * (2. * np.pi / 6))
    df['time_type_of_day_cos'] = np.cos(df.time_type_of_the_day * (2. * np.pi / 6))
    
    df['month_sin'] = np.sin((df.month - 1) * (2. * np.pi / 12))
    df['month_cos'] = np.cos((df.month - 1) * (2. * np.pi / 12))

    df['day_sin'] = np.sin((df.day - 1) * (2. * np.pi / df.days_in_month))
    df['day_cos'] = np.cos((df.day - 1) * (2. * np.pi / df.days_in_month))
    
    df['weekday_sin'] = np.sin((df.weekday - 1) * (2. * np.pi / 7))
    df['weekday_cos'] = np.cos((df.weekday - 1) * (2. * np.pi / 7))
    
    df['weekofyear_sin'] = np.sin((df.weekofyear - 1) * (2. * np.pi / 52))
    df['weekofyear_cos'] = np.cos((df.weekofyear - 1) * (2. * np.pi / 52))
  
    df['dayofyear_sin'] = np.sin((df.dayofyear - 1) * (2. * np.pi / 365))
    df['dayofyear_cos'] = np.cos((df.dayofyear - 1) * (2. * np.pi / 365))

    df['quarterofyear_sin'] = np.sin((df.quarterofyear - 1) * (2. * np.pi / 4))
    df['quarterofyear_cos'] = np.cos((df.quarterofyear - 1) * (2. * np.pi / 4))

    # Clean Up Extra Features which may not be needed
    del df['hour']
    del df['minute']
    del df['month']
    del df['day']
    del df['weekday']
    del df['days_in_month']
    del df['time_type_of_the_day']
    del df['weekofyear']
    del df['quarterofyear']    
    
# end of the function to extract date components
# Calculate NY Manhattan Distance using longitude and lattitude
# I need to add credit to the author of this fucntion from StackOverFlow. 
def haversine_np(lon1, lat1, lon2, lat2):
    
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    
    miles = 3959 * c
    
    return miles
# Now read the training set

# To process the entire data set
#train_df =  pd.read_csv('train.csv')

# To play with only subset of the data
fname = 'train.csv'
train_df =  pd.read_csv('../input/train.csv', nrows = TRAINING_SIZE)

# Let's find out if there are some null values
print(train_df.isnull().sum())

# Let's drop the null values
print(fname + ' Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print(fname + ' New size: %d' % len(train_df))
# The next couple of sections focuses on data cleaning and feature extractions which is kind of excessive for Deep Learning
# probably that's the diference between theoritical claim and practical reality...

# Important feature here two dimensional latitude and longitudes to three dimesnional coordinate systems.
add_travel_vector_features(train_df)

# Extract various date components.
parse_and_extract_date_time_components(train_df)

# NEED TO CHECK THIS VALUE - PROBABLY FIND OUT THE MAXIMUM RANGE OF NEW YORK CITY LIMITS

print(fname + ' Before dropping Zero pickup longitude and latitude - Old size: %d' % len(train_df))
train_df = train_df[(train_df.pickup_longitude != 0.) & (train_df.pickup_latitude != 0.)]
print(fname + ' After dropping Zero pickup longitude and latitude - New size: %d' % len(train_df))

print(fname + ' Before dropping Zero dropoff longitude and latitude - Old size: %d' % len(train_df))
train_df = train_df[(train_df.dropoff_longitude != 0.) & (train_df.dropoff_latitude != 0.)]
print(fname + ' After dropping Zero dropoff longitude and latitude - New size: %d' % len(train_df))

print(fname + ' Before dropping really large lat and lon Outliers - Old size: %d' % len(train_df))
train_df = train_df[(train_df.delta_longitude.abs() < 5.0) & (train_df.delta_latitude.abs() < 5.0)]
print(fname + ' After dropping really large lat and lon Outliers - New size: %d' % len(train_df))

print(fname + ' Before dropping zero delta lat and long values - Old size: %d' % len(train_df))
train_df = train_df[(train_df.delta_longitude != 0.) & (train_df.delta_latitude != 0.)]
print(fname + ' After dropping delta lat and long values - New size: %d' % len(train_df))

# ADD CODE TO CHECK FOR NEGATIVE FARES
print(fname + ' Before dropping NEGATIVE FARE VALUES - Old size: %d' % len(train_df))
train_df = train_df[(train_df.fare_amount > 0.)]
print(fname + ' After dropping NEGATIVE FARE VALUES - New size: %d' % len(train_df))

# ADD CODE TO CHECK FOR ZERO PASSENGER COUNTS
print(fname + ' Before dropping Zero Passenger Counts - Old size: %d' % len(train_df))
train_df = train_df[(train_df.passenger_count > 0) & (train_df.passenger_count < 7)]
print(fname + ' After dropping Zero Passenger Counts - New size: %d' % len(train_df))

# Extract more features for the model
train_df['distance_in_miles'] = haversine_np(train_df.dropoff_longitude, train_df.dropoff_latitude, train_df.pickup_longitude, train_df.pickup_latitude)

train_df['direction'] = calculate_direction(train_df.delta_longitude, train_df.delta_latitude)

# JFK AS THE FRAME OF REFERENCE
train_df['pickup_to_JFK_distance'] = haversine_np(train_df.pickup_longitude, train_df.pickup_latitude, JFK[0], JFK[1]) 
train_df['dropoff_to_JFK_distance'] = haversine_np(train_df.dropoff_longitude, train_df.dropoff_latitude, JFK[0], JFK[1]) 

# ALSO USE EWR AS ANOTHER FRAME OF REFERENCE IF REQUIRED
train_df['pickup_to_EWR_distance'] = haversine_np(train_df.pickup_longitude, train_df.pickup_latitude, EWR[0], EWR[1]) 
train_df['dropoff_to_EWR_distance'] = haversine_np(train_df.dropoff_longitude, train_df.dropoff_latitude, EWR[0], EWR[1]) 

# ALSO USE LGA AS ANOTHER FRAME OF REFERENCE IF REQUIRED
train_df['pickup_to_LGA_distance'] = haversine_np(train_df.pickup_longitude, train_df.pickup_latitude, LGA[0], LGA[1]) 
train_df['dropoff_to_LGA_distance'] = haversine_np(train_df.dropoff_longitude, train_df.dropoff_latitude, LGA[0], LGA[1]) 

# ALSO USE NYC AS ANOTHER FRAME OF REFERENCE IF REQUIRED
train_df['pickup_to_NYC_distance'] = haversine_np(train_df.pickup_longitude, train_df.pickup_latitude, NYC[0], NYC[1]) 
train_df['dropoff_to_NYC_distance'] = haversine_np(train_df.dropoff_longitude, train_df.dropoff_latitude, NYC[0], NYC[1]) 

# For some reason my NN model didn't find Euclidean and Manhattan distances that important.

# Calculate Manhattan Distance between two sets of GPS coordinates
train_df['manhattan'] = minkowski_distance(train_df.pickup_longitude, train_df.dropoff_longitude, 
                                           train_df.pickup_latitude, train_df.dropoff_latitude, 1)

# Calcualte Euclidean Distance between two sets of GPS Coordinates
train_df['euclidean'] = minkowski_distance(train_df.pickup_longitude, train_df.dropoff_longitude, 
                                           train_df.pickup_latitude, train_df.dropoff_latitude, 2)

# Calcualte 3D Euclidean Distance between two sets 3D GPS Coordinates
train_df['euclidean_3D'] = minkowski_distance_3D(train_df.pickup_x, train_df.dropoff_x, 
                                                 train_df.pickup_y, train_df.dropoff_y, 
                                                 train_df.pickup_z, train_df.dropoff_z, 2)

# Calcualte 3D Manhattan Distance between two sets 3D GPS Coordinates
train_df['manhattan_3D'] = minkowski_distance_3D(train_df.pickup_x, train_df.dropoff_x, 
                                                 train_df.pickup_y, train_df.dropoff_y, 
                                                 train_df.pickup_z, train_df.dropoff_z, 1)

"""
# JFK AS THE EUCLIDEAN FRAME OF REFERENCE
train_df['euclidean_pickup_to_JFK'] = minkowski_distance(train_df.pickup_longitude, JFK[0], train_df.pickup_latitude,  JFK[1], 2) 
train_df['euclidean_dropoff_to_JFK'] = minkowski_distance(train_df.dropoff_longitude, JFK[0], train_df.dropoff_latitude, JFK[1], 2) 

# NYC AS THE EUCLIDEAN FRAME OF REFERENCE
train_df['euclidean_pickup_to_NYC'] = minkowski_distance(train_df.pickup_longitude, NYC[0], train_df.pickup_latitude,  NYC[1], 2) 
train_df['euclidean_dropoff_to_NYC'] = minkowski_distance(train_df.dropoff_longitude, NYC[0], train_df.dropoff_latitude, NYC[1], 2) 

# EWR AS THE EUCLIDEAN FRAME OF REFERENCE
train_df['euclidean_pickup_to_EWR'] = minkowski_distance(train_df.pickup_longitude, EWR[0], train_df.pickup_latitude,  EWR[1], 2) 
train_df['euclidean_dropoff_to_EWR'] = minkowski_distance(train_df.dropoff_longitude, EWR[0], train_df.dropoff_latitude, EWR[1], 2) 
"""

# ADD CODE TO CHECK FOR REALLY HIGH FARE AMOUNTS
print(fname + ' Before dropping HIGH FARE AMOUNTS - Old size: %d' % len(train_df))
train_df = train_df[(train_df.fare_amount > 1.) & (train_df.fare_amount < 251.)]
print(fname + ' After dropping HIGH FARE AMOUNTS - New size: %d' % len(train_df))

# Extract the training features and standardize the values
training_set = train_df.loc[:, [
                                'passenger_count',
                                'distance_in_miles',
                                'pickup_x',
                                'pickup_y',
                                'pickup_z',
                                'dropoff_x',
                                'dropoff_y',
                                'dropoff_z',
                                'pickup_to_JFK_distance',
                                'dropoff_to_JFK_distance',
                                'pickup_to_NYC_distance',
                                'dropoff_to_NYC_distance',
                                'delta_longitude',
                                'delta_latitude',
                                'direction',
                                'year',
                                'linear_time',
                                'peak_hours',
                                'surcharge',
                                'hour_sin',
                                'hour_cos',
                                'minute_sin',
                                'minute_cos',
                                'time_type_of_day_sin',
                                'time_type_of_day_cos',
                                'month_sin',
                                'month_cos',
                                'day_sin',
                                'day_cos',
                                'weekday_sin',
                                'weekday_cos',
                                'weekofyear_sin',
                                'weekofyear_cos',
                                'quarterofyear_sin',
                                'quarterofyear_cos'
                                ]].values

# Extract the training label
training_label = train_df.loc[:, 'fare_amount'].values

# Standardize the training data
mean = training_set.mean(axis = 0)

training_set -= mean

std = training_set.std(axis = 0)

training_set /= std # this corresponds to sci kit fit & transform
    
# Code to drop the columns if required - please use the second set of indexes
# training_set = np.delete(training_set, [21, 22, 23, 24, 27, 28], axis = 1)

# ADD CODE TO GENERATE VALIDATION SET
# define records to be considered for validation
num_validation_samples = VALIDATION_SIZE

# Check if sometimes Validation Set is larger than training set:
if num_validation_samples >= len(training_set):
    # Set the validation data set to 20% of training set
    num_validation_samples = int(len(training_set) * 0.2)
# end of if

# Validation Data for input
validation_data = training_set[:num_validation_samples]
training_set = training_set[num_validation_samples:]

# Validation Data for target
validation_targets = training_label[:num_validation_samples]
training_label = training_label[num_validation_samples:]

# BEGIN MODEL DEFINITION

# Code snippet to load pretrained model and save weights
#model = load_model('./GoodModelRepo/NYCTFP_Model_4.h5')
#model.save_weights('./GoodModelRepo/NYCTFP_weights_4.h5')
#del model

#kernel_regularizer = regularizers.l2(0.001)
#kernel_regularizer = regularizers.l2(0.0001)
#kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001)
#kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001)

NN_INPUT_SIZE = training_set.shape[1]

if preTrainedModel == True:
    
    # Load from the previous best model
    model = load_model('./GoodModelRepo/0_NYCTFP_Model_8.h5')
    
else:
    
    # Intialize the model
    model = models.Sequential()
    
    #model.add(layers.Dense(128, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu', input_shape = (NN_INPUT_SIZE,)))
    model.add(layers.Dense(128, kernel_initializer = 'uniform', activation = 'relu', input_shape = (NN_INPUT_SIZE,)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    # Add a second layer
    #model.add(layers.Dense(128, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
    model.add(layers.Dense(128, kernel_initializer = 'uniform', activation = 'relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    # Add a third layer
    #model.add(layers.Dense(64, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
    model.add(layers.Dense(64, kernel_initializer = 'uniform', activation = 'relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    # Add a fourth layer
    #model.add(layers.Dense(32, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
    model.add(layers.Dense(32, kernel_initializer = 'uniform', activation = 'relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    """
    # Add a fifth layer
    #model.add(layers.Dense(8, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
    model.add(layers.Dense(16, kernel_initializer = 'uniform', activation = 'relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    # Add a fifth layer
    #model.add(layers.Dense(8, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
    model.add(layers.Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    
    """
    
    # Add a output layer without an activation function
    model.add(layers.Dense(1))
    
    #model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    #model.compile(optimizer = optimizers.Adam(lr=0.01), loss = 'mse', metrics = ['mae'])
    
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    #model.compile(optimizer = optimizers.RMSprop(lr=0.01), loss = 'mse', metrics = ['mae'])
    #model.compile(optimizer = optimizers.RMSprop(lr=0.0006), loss = 'mse', metrics = ['mae'])
    
    #model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mae'])
    
    # Load the good weights that we got from previous best score
    # These weights are good for 64/64/1 architecture
    # FOR NEW ARCHITECTURES THE FOLLOWING LINE SHOULD BE COMMENTED
    if loadOnlyWeights == True:    
        model.load_weights('./GoodModelRepo/0_NYCTFP_Weights_8.h5')

# END OF MODEL DEFINITION

# DEFINE A CALLBACK TO SAVE THE BEST MODEL WITH MINIMUM VALIDATION MAE VALUE
# keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpointer = ModelCheckpoint(filepath = 'NYCTFP_Ref_Model.h5', monitor = 'val_mean_absolute_error', verbose = 1, save_best_only = True, mode = 'min')

# Train the model
test_history = model.fit(training_set, training_label, shuffle=True,
                         validation_data = (validation_data, validation_targets), 
                         epochs = EPOCH, batch_size = BATCH_SIZE,
                         callbacks = [checkpointer])

print(os.listdir("/kaggle/working")) #print(os.getcwd()) #print(os.listdir(".")) # /kaggle/working is the current directory
# PLOT THE GRAPH HERE TO TRACE MAE AND LOSS.

# Code to Evaluate the model on the validation data - if required
#val_mse, val_mae = model.evaluate(validation_data, validation_targets)
#print(val_mse, val_mae)

if INTEL_AI_DEV_CLOUD == False:
    
    # Draw the graphs on the MacBook
    
    # forms set of Y axis values
    mae = test_history.history['mean_absolute_error']
    val_mae = test_history.history['val_mean_absolute_error']
    loss = test_history.history['loss']
    val_loss = test_history.history['val_loss']

    # forms X axis values
    epochs = range(1, len(mae) + 1)

    # Process Training and Validation Accuracy
    plt.plot(epochs, mae, 'bo', label = 'New York Taxi Fare Training MAE')
    plt.plot(epochs, val_mae, 'r', label = 'New York Taxi Fare Validation MAE')
    plt.title('New York Taxi Fare Training and Validation MAE Comparison')
    plt.legend()
    plt.show()

    # Process Training and Validation Losses
    plt.plot(epochs, loss, 'ro', label = 'New York Taxi Fare Training Loss')
    plt.plot(epochs, val_loss, 'g', label = 'New York Taxi Fare Validation Loss')
    plt.title('New York Taxi Fare Training and Validation Loss Comparison')
    plt.legend()
    plt.show()

# end of if statement

# Always save the model after training - we could reuse it as the sophistication increases.
model.save('NYCTFP_Model_12.h5')
model.save_weights('NYCTFP_Weights_12.h5')
pd.DataFrame(test_history.history).to_csv("epoch_testing_history.csv", header = True, index = False)

# Release the model created for training. We will create a new one for testing.
# del model

print(os.listdir("/kaggle/working")) #print(os.getcwd()) #print(os.listdir(".")) # /kaggle/working is the current directory
# Read the test file
test_df = pd.read_csv('../input/test.csv')

print(test_df.isnull().sum())

# Let's drop the null values - THERE SHOULDN'T BE ANY...
print('test.csv Old size: %d' % len(test_df))
test_df = test_df.dropna(how = 'any', axis = 'rows')
print('test.csv New size: %d' % len(test_df))

# Generate the necessary features for test data file
add_travel_vector_features(test_df)

parse_and_extract_date_time_components(test_df)

# Generate more features - we need to match the input layout provided to train the neural network

# Calculate the distance in miles
test_df['distance_in_miles'] = haversine_np(test_df.dropoff_longitude, test_df.dropoff_latitude, test_df.pickup_longitude, test_df.pickup_latitude)

test_df['direction'] = calculate_direction(test_df.delta_longitude, test_df.delta_latitude)

# JFK AS THE FRAME OF REFERENCE
test_df['pickup_to_JFK_distance'] = haversine_np(test_df.pickup_longitude, test_df.pickup_latitude, JFK[0], JFK[1]) 
test_df['dropoff_to_JFK_distance'] = haversine_np(test_df.dropoff_longitude, test_df.dropoff_latitude, JFK[0], JFK[1]) 

# ALSO USE EWR AS ANOTHER FRAME OF REFERENCE IF REQUIRED
test_df['pickup_to_EWR_distance'] = haversine_np(test_df.pickup_longitude, test_df.pickup_latitude, EWR[0], EWR[1]) 
test_df['dropoff_to_EWR_distance'] = haversine_np(test_df.dropoff_longitude, test_df.dropoff_latitude, EWR[0], EWR[1]) 

# ALSO USE LGA AS ANOTHER FRAME OF REFERENCE IF REQUIRED
test_df['pickup_to_LGA_distance'] = haversine_np(test_df.pickup_longitude, test_df.pickup_latitude, LGA[0], LGA[1]) 
test_df['dropoff_to_LGA_distance'] = haversine_np(test_df.dropoff_longitude, test_df.dropoff_latitude, LGA[0], LGA[1]) 

# ALSO USE NYC AS ANOTHER FRAME OF REFERENCE IF REQUIRED
test_df['pickup_to_NYC_distance'] = haversine_np(test_df.pickup_longitude, test_df.pickup_latitude, NYC[0], NYC[1]) 
test_df['dropoff_to_NYC_distance'] = haversine_np(test_df.dropoff_longitude, test_df.dropoff_latitude, NYC[0], NYC[1]) 

# Calculate Manhattan Distance between two sets of GPS coordinates
test_df['manhattan'] = minkowski_distance(test_df.pickup_longitude, test_df.dropoff_longitude, 
                                          test_df.pickup_latitude, test_df.dropoff_latitude, 1)

# Calcualte Euclidean Distance between two sets of GPS Coordinates
test_df['euclidean'] = minkowski_distance(test_df.pickup_longitude, test_df.dropoff_longitude, 
                                          test_df.pickup_latitude, test_df.dropoff_latitude, 2)

# Calcualte 3D Euclidean Distance between two sets 3D GPS Coordinates
test_df['euclidean_3D'] = minkowski_distance_3D(test_df.pickup_x, test_df.dropoff_x, 
                                                 test_df.pickup_y, test_df.dropoff_y, 
                                                 test_df.pickup_z, test_df.dropoff_z, 2)

# Calcualte 3D Manhattan Distance between two sets 3D GPS Coordinates
test_df['manhattan_3D'] = minkowski_distance_3D(test_df.pickup_x, test_df.dropoff_x, 
                                                 test_df.pickup_y, test_df.dropoff_y, 
                                                 test_df.pickup_z, test_df.dropoff_z, 1)
"""
# JFK AS THE EUCLIDEAN FRAME OF REFERENCE
test_df['euclidean_pickup_to_JFK'] = minkowski_distance(test_df.pickup_longitude, JFK[0], test_df.pickup_latitude,  JFK[1], 2) 
test_df['euclidean_dropoff_to_JFK'] = minkowski_distance(test_df.dropoff_longitude, JFK[0], test_df.dropoff_latitude, JFK[1], 2) 

# NYC AS THE EUCLIDEAN FRAME OF REFERENCE
test_df['euclidean_pickup_to_NYC'] = minkowski_distance(test_df.pickup_longitude, NYC[0], test_df.pickup_latitude,  NYC[1], 2) 
test_df['euclidean_dropoff_to_NYC'] = minkowski_distance(test_df.dropoff_longitude, NYC[0], test_df.dropoff_latitude, NYC[1], 2) 

# EWR AS THE EUCLIDEAN FRAME OF REFERENCE
test_df['euclidean_pickup_to_EWR'] = minkowski_distance(test_df.pickup_longitude, EWR[0], test_df.pickup_latitude,  EWR[1], 2) 
test_df['euclidean_dropoff_to_EWR'] = minkowski_distance(test_df.dropoff_longitude, EWR[0], test_df.dropoff_latitude, EWR[1], 2) 

"""
print('Done with Test Data File Feature Generations')
# Extract the testing features to be used for extraction - it needs to match with features used for training.

testing_set = test_df.loc[:, [
                                'passenger_count',
                                'distance_in_miles',
                                'pickup_x',
                                'pickup_y',
                                'pickup_z',
                                'dropoff_x',
                                'dropoff_y',
                                'dropoff_z',
                                'pickup_to_JFK_distance',
                                'dropoff_to_JFK_distance',
                                'pickup_to_NYC_distance',
                                'dropoff_to_NYC_distance',
                                'delta_longitude',
                                'delta_latitude',
                                'direction',
                                'year',
                                'linear_time',
                                'peak_hours',
                                'surcharge',
                                'hour_sin',
                                'hour_cos',
                                'minute_sin',
                                'minute_cos',
                                'time_type_of_day_sin',
                                'time_type_of_day_cos',
                                'month_sin',
                                'month_cos',
                                'day_sin',
                                'day_cos',
                                'weekday_sin',
                                'weekday_cos',
                                'weekofyear_sin',
                                'weekofyear_cos',
                                'quarterofyear_sin',
                                'quarterofyear_cos'
                                ]].values

# Now standardize the test data using mean and standard deviation calculated using training data
testing_set -= mean
testing_set /= std # this corresponds to sci kit transform

# Code to drop the columns if required - please use the second set of indexes
# testing_set = np.delete(testing_set, [21, 22, 23, 24, 27, 28], axis = 1)

# Create the NN Model for prediction
if useBestModel == True:
    
    # Then use the best model to score test data
    del model
    model = load_model('NYCTFP_Ref_Model.h5')
    
    # testModel = load_model('NYCTFP_Ref_Model.h5')
# end of if useBest Model

# TEST PREDICTION HERE AND GENERATE OUTPUT FILE
predictions = model.predict(testing_set) #, batch_size = BATCH_SIZE)

outputKeys = np.reshape(test_df.iloc[:, 0].values, (len(test_df), 1))

outputdata = np.concatenate((outputKeys, predictions), axis = 1)

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(outputdata, columns = ['key', 'fare_amount']) #, index = outputdata[:, 0])
# submission = pd.DataFrame({'key': outputKeys, 'fare_amount': predictions}, columns = ['key', 'fare_amount'], index = test_df.iloc[:, 0] )

submission.to_csv('submission.csv', index = False)

#del model
print(os.listdir("/kaggle/working")) #print(os.getcwd()) #print(os.listdir(".")) # /kaggle/working is the current directory
#import subprocess
#subprocess.run('kaggle competitions submit -c new-york-city-taxi-fare-prediction -f /kaggle/working/submission.csv -m "Submitted From Kaggle Notebook"')
#kaggle competitions submit -c new-york-city-taxi-fare-prediction -f submission.csv -m "Submitted From Kaggle Notebook"