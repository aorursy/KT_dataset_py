import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
import s2sphere
from calendar import timegm

from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import regularizers

# from sklearn.model_selection import train_test_split
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600
DAYS_PER_WEEK = 7

SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_WEEK = SECONDS_PER_DAY * DAYS_PER_WEEK

# coarse modeling of bottlenecks in space
S2CELL_DEPTH = 11

def pickup_time(row):
    utc_time = time.strptime(row["pickup_datetime"], "%Y-%m-%d %H:%M:%S UTC")
    return utc_time

def pickup_epoch(row):
    return timegm(row["pickup_time"])

def year(row):
    return row["pickup_time"].tm_year

def month(row):
    return row["pickup_time"].tm_mon

def day(row):
    return row["pickup_time"].tm_mday

def tod(row):
    tod_sec = row["pickup_epoch"] % SECONDS_PER_DAY
    tod_hr = int(tod_sec // SECONDS_PER_HOUR)
    return tod_hr

def tow_hr(row):
    tow_sec = row["pickup_epoch"] % SECONDS_PER_WEEK
    tow_hr = int(tow_sec // SECONDS_PER_HOUR)
    return tow_hr

def tow_day(row):
    tow_sec = row["pickup_epoch"] % SECONDS_PER_WEEK
    tow_day = int(tow_sec // SECONDS_PER_DAY)
    return tow_day

def distance(row):
    lat1 = row["pickup_latitude"]
    lon1 = row["pickup_longitude"]
    lat2 = row["dropoff_latitude"]
    lon2 = row["dropoff_longitude"]
    radius = 6371 # kilometers

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d # int(np.nan_to_num(np.round(d, 0)))

def pickup_cell_id(row):
    geohash = ''
    try:
        geohash =  s2sphere.CellId.from_lat_lng(
            s2sphere.LatLng.from_degrees(lat=row["pickup_latitude"], lng=row["pickup_longitude"])
        ).parent(S2CELL_DEPTH).to_token()
    except:
        pass
    return geohash

def dropoff_cell_id(row):
    geohash = ''
    try:
        geohash = s2sphere.CellId.from_lat_lng(
            s2sphere.LatLng.from_degrees(lat=row["dropoff_latitude"], lng=row["dropoff_longitude"])
        ).parent(S2CELL_DEPTH).to_token()
    except:
        pass
    return geohash

def enrich(df):
    df["pickup_time"] = df.apply(pickup_time, axis=1)
    df["pickup_epoch"] = df.apply(pickup_epoch, axis=1)
    df["year"] = df.apply(year, axis=1)
    df["month"] = df.apply(month, axis=1)
    df["day"] = df.apply(day, axis=1)
    df["tod"] = df.apply(tod, axis=1)
    df["tow_hr"] = df.apply(tow_hr, axis=1)
    df["tow_day"] = df.apply(tow_day, axis=1)
    df["distance"] = df.apply(distance, axis=1)
    df["pickup_cell_id"] = df.apply(pickup_cell_id, axis=1)
    df["dropoff_cell_id"] = df.apply(dropoff_cell_id, axis=1)
    return df

def clean_dataset(x):
    return x[(x.fare_amount>0)& (x.pickup_longitude !=0) & (x.pickup_latitude !=0) & 
            (x.dropoff_longitude !=0) &(x.dropoff_latitude !=0)]

def process(df, train=True):
    if train:
        #drop empty cells
        df = df.dropna(how='any', axis='rows')
        #removing zeros 
        df = clean_dataset(df)
    return enrich(df)
    
    
cat_cols = ['pickup_cell_id', 'dropoff_cell_id', 'passenger_count', 'tod', 
            'tow_hr', 'tow_day', 'year', 'month', 'day']
feature_cols = cat_cols + ['distance']
label_col = 'fare_amount'
# Lets read 1M rows to define categorical features
MIN_FEAT_REP = 10 # out of 1M rows the feature should at least repeat 10 times
df = pd.read_csv('../input/train.csv', nrows=1_000_000)
df = process(df)

COUNT_FEAT = {}
for feat in cat_cols:
    COUNT_FEAT[feat] = {}
    for val in df[feat]:
        COUNT_FEAT[feat][val] = COUNT_FEAT[feat].setdefault(val, 0) + 1

idx = 0
FEAT_TO_IDX = {}
for feat in cat_cols:
    FEAT_TO_IDX[feat] = {}
    for val in df[feat].unique():
        if COUNT_FEAT[feat][val] >= MIN_FEAT_REP:
            FEAT_TO_IDX[feat][val] = idx
            idx += 1

NUM_CAT_FEATS = idx
NUM_FEATS = NUM_CAT_FEATS + 1 # distance is not treated as categorical

del df
import gc; gc.collect()
print (NUM_FEATS)
def get_labeled_feats(df):
    if label_col in df.columns: 
        y = df[label_col].values 
    else:
        y = None
    X = np.zeros((df.shape[0], NUM_FEATS))
    for i, irow in enumerate(df.iterrows()):
        _, row = irow
        X[i, -1] = row["distance"]
        for feat in cat_cols:
            idx = FEAT_TO_IDX[feat].get(row[feat])
            if idx is not None: X[i, idx] = 1.
    return np.nan_to_num(X), np.nan_to_num(y)
inp = Input(shape=(NUM_FEATS,))
x = Dense(128, activation="linear", kernel_regularizer=regularizers.l2(0.001))(inp)
x = Dense(64, activation="elu")(x)
x = Dense(32, activation="elu")(x)
x = Dense(16, activation="elu")(x)
x = Dense(1, activation="relu")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
chunksize = 500_000
def process_fully(df, num_epochs=10):
    df = process(df)
    X, y = get_labeled_feats(df)
    model.fit(X, y, batch_size=512, epochs=num_epochs)
    
    
test_df = process(pd.read_csv('../input/test.csv'), train=False)
test_X, _ = get_labeled_feats(test_df)
for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize):
    process_fully(chunk)
    test_y_pred = model.predict(test_X, batch_size=1024)[:, 0]
    out_df = pd.DataFrame({"key": test_df["key"].values})
    out_df['fare_amount'] = test_y_pred
    out_df.to_csv("submission.csv", index=False)
