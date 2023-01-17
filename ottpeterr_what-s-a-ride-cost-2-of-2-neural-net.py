import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
# thanks to szelee for this quick loading method
p = subprocess.Popen(['wc', '-l', TRAIN_PATH], stdout=subprocess.PIPE, 
                                               stderr=subprocess.PIPE)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err)
n_rows = int(result.strip().split()[0])+1
# Params 
FARE_MIN = 2.00
FARE_MAX = 150
YEAR_MIN = 2000
YEAR_MAX = 2018
PASSENGER_MIN = 1
PASSENGER_MAX = 6
LAT_MIN  = 39.9
LAT_MAX  = 41.3
LONG_MIN = -74.4
LONG_MAX = -72.5
DIST_MIN = 0.05
DIST_MAX = 35

# Bin params
NUM_LAT_BINS = 140*5
NUM_LONG_BINS = 190*5
NUM_TIME_BINS = 480
NUM_DIST_BINS = 400

lat_bins = np.linspace(LAT_MIN, LAT_MAX, NUM_LAT_BINS+1).tolist()
lat_bins = [-90] + lat_bins + [90]
long_bins = np.linspace(LONG_MIN, LONG_MAX, NUM_LONG_BINS+1).tolist()
long_bins = [-180] + long_bins + [180]
time_bins = np.linspace(-1, 2400, NUM_TIME_BINS+1).tolist()
dist_bins = np.linspace(DIST_MIN-1, DIST_MAX+1, NUM_DIST_BINS+1).tolist()
# thanks to madhurisivalenka for this function
def add_haversine_distance_feature(df, lat1='pickup_latitude', long1='pickup_longitude', lat2='dropoff_latitude', long2='dropoff_longitude'):
    #R = 6371  # radius of earth in kilometers
    R = 3959 # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c
    d = (R * c)
    df["distance"] = d.astype('float32')
    
    return df
def drop_conditional(df):
    # 1.
    df = df.drop(df[df.isnull().any(1)].index, axis=0)
    df = df.drop(df[df.isin([np.nan, np.inf, -np.inf]).any(1)].index, axis=0)
    with pd.option_context('mode.use_inf_as_null', True):
        df = df.dropna()

    # 2.
    df = df.drop(df[df.fare_amount > FARE_MAX].index, axis=0)
    df = df.drop(df[df.fare_amount < FARE_MIN].index, axis=0)
    
    # 3.
    df = df.drop(df[df.passenger_count > PASSENGER_MAX].index, axis = 0)
    df = df.drop(df[df.passenger_count < PASSENGER_MIN].index, axis = 0)

    # 4.
    df = df.drop(df[df.pickup_latitude > LAT_MAX].index, axis=0)
    df = df.drop(df[df.pickup_latitude < LAT_MIN].index, axis=0)
    
    df = df.drop(df[df.pickup_longitude > LONG_MAX].index, axis=0)
    df = df.drop(df[df.pickup_longitude < LONG_MIN].index, axis=0)

    df = df.drop(df[df.dropoff_latitude > LAT_MAX].index, axis=0)
    df = df.drop(df[df.dropoff_latitude < LAT_MIN].index, axis=0)
    
    df = df.drop(df[df.dropoff_longitude > LONG_MAX].index, axis=0)
    df = df.drop(df[df.dropoff_longitude < LONG_MIN].index, axis=0)
    
    # 5.
    df = df.drop(df[df.distance > DIST_MAX].index, axis = 0)
    df = df.drop(df[df.distance < DIST_MIN].index, axis = 0)
    
    df = df.drop(df[df.year > YEAR_MAX].index, axis = 0)
    df = df.drop(df[df.year < YEAR_MIN].index, axis = 0)
    
    return df
def add_date_features(df):
    # 6.
    # df.drop(columns=['key'], inplace=True)
    df.pickup_datetime = df.pickup_datetime.str.slice(0, 16)
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime, utc=True, format='%Y-%m-%d %H:%M')
    # 7.
    df['year'] = df.pickup_datetime.dt.year.astype('uint16')
    df['month'] = df.pickup_datetime.dt.month.astype('uint8')
    # df['day'] = df.pickup_datetime.dt.day.astype('uint8')
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek.astype('uint8')
    hours = df.pickup_datetime.dt.hour.astype('uint8')
    minutes = df.pickup_datetime.dt.minute.astype('uint8')
    # combine the minutes and hours into a single variable
    df['time_of_day'] = (hours*100.0) + (minutes*(5.0/3)) # makes time [0 - 2399]
    # don't need the pickup_datetime anymore since it's been divided into the above cols
    df = df.drop(columns=['pickup_datetime'])
    
    return df
def bin_data(df):
    df.pickup_latitude = pd.cut(df.pickup_latitude, precision=7, bins=lat_bins, labels=False).astype("uint16")
    df.pickup_longitude = pd.cut(df.pickup_longitude, precision=7, bins=long_bins, labels=False).astype("uint16")
    df.dropoff_latitude = pd.cut(df.dropoff_latitude, precision=7, bins=lat_bins, labels=False).astype("uint16")
    df.dropoff_longitude = pd.cut(df.dropoff_longitude, precision=7, bins=long_bins, labels=False).astype("uint16")
    df.time_of_day = pd.cut(df.time_of_day, bins=time_bins, precision=7, labels=False).astype("uint16")
    df.distance = pd.cut(df.distance, bins=dist_bins, precision=7, labels=False).astype("uint16")
    return df
def clean_data(df, test=False):
    df = add_haversine_distance_feature(df)
    df = add_date_features(df)
    if not test:
        df = drop_conditional(df) 
        df = bin_data(df)
    return df
traintypes = {'fare_amount': 'float16',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
chunksize = 2**20
total_chunk = n_rows // chunksize + 1

def load_all_data(X):
    df_list = [] # list to hold the batch dataframe
    i = 0
    limit = -1
    for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize):    
        i = i+1
        # Each chunk is a corresponding dataframe
        print(f'DataFrame Chunk {i:02d}/{total_chunk}')
        df_chunk = clean_data(df_chunk)
        # Alternatively, append the chunk to list and merge all
        df_list.append(df_chunk)
        if i == limit:
            break
    
    X = pd.concat(df_list)
    return X
# Shuffle
def shuffle(X):
    X = X.sample(frac=1).reset_index(drop=True)
    return X
# Cut down on the data size
def reduce_data(X):
    drop_portion = 0.25
    X = X.drop(X.index[0:int(X.shape[0]*drop_portion)])
    return X
# Normalize
# also change data into different float types to save on memory where we can
minmax = pd.DataFrame()
float32cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance", "time_of_day"]
float16cols = ["passenger_count", "month", "dayofweek", "year"]

def normalize(X):
    for col in float32cols:
        col_min = X[col].min()
        col_max = X[col].max()
        minmax[col] = (col_min, col_max)
        X[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float32')

    for col in float16cols:
        col_min = X[col].min()
        col_max = X[col].max()
        minmax[col] = (col_min, col_max)
        X[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float16')

    minmax.time_of_day = (-1, 2400)
    return X
#     print(X.head())
#     print(X.info())
# actually run everything
X = pd.DataFrame()
X = load_all_data(X)
X.info()
print("Loading complete")
X = shuffle(X)
print("Shuffle complete")
X = reduce_data(X)
print("Reduction complete")
X = normalize(X)
print("Normalization complete")
print(X.head())
print(X.info())
# take out the answers
y = []
y = X.fare_amount
X = X.drop(columns="fare_amount")
# Splitting off a validation set
validation_portion = 1.0/1001
index = int(X.shape[0]*validation_portion)
print("training:\t%d\nvalidation:\t%d" % (X.shape[0]-index, index))
val_X = X[0:index]
X = X.drop(X.index[0:index])
val_y = y[0:index]
y = y.drop(y.index[0:index])
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import metrics
from keras import backend as K
K.set_image_dim_ordering('tf')

# Using a GPU for this kernel
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
model = Sequential()

model.add(Dense(32, kernel_initializer="normal", input_dim=X.shape[1], activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5));
model.add(Dense(32, kernel_initializer="normal", activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer="normal", activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer="normal", activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='nadam', 
              metrics=[metrics.mae])
num_epochs = 10
batch_size = 2**9
history = model.fit(X.values, y.values, 
                    validation_data = (val_X, val_y),
                    shuffle=True, 
                    epochs=num_epochs, 
                    batch_size=batch_size)
plt.figure()
plt.plot(history.history['loss'], color="blue")
plt.plot(history.history['val_loss'], color="red")
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylabel("loss")
plt.xlabel("epoch")

plt.figure()
plt.plot(history.history['mean_absolute_error'], color="blue")
plt.plot(history.history['val_mean_absolute_error'], color="red")
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylabel("Mean Abs. Error")
plt.xlabel("epoch")
# a little test
val_pred = model.predict(val_X[0:5]).flatten()
print("actual: "+str(val_y[0:5].values))
print("pred:   "+str(val_pred))
traintypes = {'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
cols.append('key')

X_test = pd.read_csv(TEST_PATH, usecols=cols, dtype=traintypes)
X_test = clean_data(X_test, test=True)
X_test_key = X_test['key']
X_test.drop(columns=['key'], inplace=True)

# normalize with the same values as the train data

for col in float16cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float16')

for col in float32cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float32')
    
X_test.head()
pred = model.predict(X_test).flatten()
pred = np.round(pred,2)
results = pd.DataFrame({'key': X_test_key, 'fare_amount': pred})
results.key = results.key.astype(str)
results.info()
results.to_csv('submission.csv', index=False)
print(results[0:5])