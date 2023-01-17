import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 16,9
from tqdm import tqdm
import os
def fast_csv_sampler(path, n, chunk_size=32):
    fsize = os.path.getsize(path)
    names = pd.read_csv(path, nrows=0).columns

    with open(path, 'r') as f:
        while True:
            dfs = []

            for i in range(n // chunk_size):
                f.seek(np.random.randint(0, fsize))
                f.readline()
                dfs.append(pd.read_csv(f, nrows=chunk_size, names=names))

            df = pd.concat(dfs, ignore_index=True)
            df['pickup_datetime'] = df['pickup_datetime'].apply(pd.Timestamp).dt.tz_convert(None)

            yield df
def prep_data(df, shuffle=False):
    X_cat = np.vstack([
        df['pickup_datetime'].dt.hour, # 0-23
        df['pickup_datetime'].dt.weekday + 24, # 24-30
        df['pickup_datetime'].dt.dayofyear + 30, # 31-396,
        df['pickup_datetime'].dt.weekofyear + 396, # 397-449,
        df['pickup_datetime'].dt.year - 2009 + 450, # 450-456
        df['passenger_count'] + 456 # 457-463
    ]).T
    
    X_deg = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].values
    X_deg /= 180
    
    X_num = np.vstack([
        ((df['pickup_datetime'] - pd.Timestamp('2009-01-01')) / (pd.Timestamp('2015-07-01') - pd.Timestamp('2009-01-01'))).values
    ]).T
    
    if 'fare_amount' in df.columns:
        y = df['fare_amount'].values
    
    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_deg = X_deg[rnd_ind]
        X_cat = X_cat[rnd_ind]
        X_num = X_num[rnd_ind]
        
        if 'fare_amount' in df.columns:
            y = y[rnd_ind]
        
    if 'fare_amount' in df.columns: 
        return [X_deg, X_cat, X_num], y

    return [X_deg, X_cat, X_num]
def prep_data_gen(df_gen, shuffle=False):
    for df in df_gen:
        df.dropna(inplace=True)
        
        is_weird = (df['fare_amount'] < 0)
        is_weird |= ~df['pickup_latitude'].between(40, 42)
        is_weird |= ~df['pickup_longitude'].between(-75, -72)
        is_weird |= ~df['dropoff_latitude'].between(40, 42)
        is_weird |= ~df['dropoff_longitude'].between(-75, -72)
        is_weird |= (df['passenger_count'] == 0)
        
        df = df[~is_weird]
        
        yield prep_data(df, shuffle=shuffle)
import keras.layers as lyr
import keras.activations as act
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping
def K_haversine_bearing(x):
    R = 6371
    
    x_rad = x * np.pi
    
    lat1, lng1, lat2, lng2 = x_rad[:,0], x_rad[:,1], x_rad[:,2], x_rad[:,3]
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = K.sin(dlat/2) * K.sin(dlat/2) + K.cos(lat1) * K.cos(lat2) * K.sin(dlng/2) * K.sin(dlng/2);
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1-a))
    
    d = K.log((R * c) + 1)
    
    x = K.sin(dlng) * K.cos(lat2)
    y = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x, y) / np.pi
    
    return K.concatenate([K.reshape(d, (-1, 1)), K.reshape(b, (-1, 1))])
lyr_latlng_input = lyr.Input((4,))
lyr_haversine_bearing = lyr.Lambda(K_haversine_bearing, name='haversine')(lyr_latlng_input)

lyr_cat_input = lyr.Input((6,))
lyr_cat_embeddings = lyr.Embedding(464, 8)(lyr_cat_input)
lyr_cat_flatten = lyr.Flatten()(lyr_cat_embeddings)

lyr_num_input = lyr.Input((1,))

lyr_concat = lyr.concatenate([lyr_latlng_input, lyr_haversine_bearing, lyr_cat_flatten, lyr_num_input])

lyr_dense1 = lyr.Dense(512, activation=act.selu)(lyr_concat)
lyr_dense2 = lyr.Dense(128, activation=act.selu)(lyr_dense1)
lyr_dense3 = lyr.Dense(32, activation=act.selu)(lyr_dense2)
lyr_out = lyr.Dense(1, activation=act.selu)(lyr_dense3)

model = Model([lyr_latlng_input, lyr_cat_input, lyr_num_input], lyr_out)
model.summary()
for i in range(8):
    model.compile(loss='mse', optimizer='nadam')
    model.fit_generator(prep_data_gen(fast_csv_sampler('../input/train.csv', 512, chunk_size=16)), steps_per_epoch=128, epochs=16, 
                        workers=2, use_multiprocessing=True)
df_test = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime'])
df_test.info()
y_test = np.clip(model.predict(prep_data(df_test), batch_size=512, verbose=1), 0, None)
sns.distplot(np.log1p(y_test))
df_sub = pd.DataFrame({
    'key': df_test['key'].values,
    'fare_amount': y_test[:,0]
}).set_index('key')
df_sub.head()
df_sub.to_csv('submission.csv')

