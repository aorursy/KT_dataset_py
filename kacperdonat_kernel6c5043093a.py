import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend
import numpy as np
from math import exp


def init_session():
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(0)
    sess=tf.Session(config=tf_config)

    keras.backend.set_session(sess)

init_session()

train_raw=pd.read_pickle("/kaggle/input/etipgdla2020c1/train.pkl")
train_raw.head(5)
train_raw.describe()
NY_latitude  = 40.7128
NY_longitude = -74.0060

# used to later cleanup
max_deviation = 2
max_distance  = 1000
def distance(lat1, lon1, lat2, lon2, **kwarg):
    R = 6371.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def prepare_features(df):
    dtcol=pd.DatetimeIndex(df['pickup_datetime'])  
    
    df['year']=dtcol.year - 2000
    df['hour']=dtcol.hour / 23
    df['day_of_week']=dtcol.dayofweek / 6
    df['day_of_year']=dtcol.dayofyear / 366
    df['week_of_year']=dtcol.weekofyear / 53
    df['diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    
    df['pickup_latitude']=df['pickup_latitude'] - NY_latitude
    df['dropoff_latitude']=df['dropoff_latitude'] - NY_latitude    
    df['pickup_longitude']=df['pickup_longitude'] - NY_longitude
    df['dropoff_longitude']=df['dropoff_longitude'] - NY_longitude
                      
    df['distance'] = distance(
        df.pickup_latitude, df.pickup_longitude, 
        df.dropoff_latitude, df.dropoff_longitude
    )
    
    df['pickup_distance_to_centre'] = distance(
        df.pickup_latitude, df.pickup_longitude, 
        0, 0
    )
    
    df['dropoff_distance_to_centre'] = distance(
        df.dropoff_latitude, df.dropoff_longitude, 
        0, 0
    )
    
    df['distence_difference'] = df['dropoff_distance_to_centre'] - df['pickup_distance_to_centre']

    #since we designed more valuable feature we will not need original timestamp
    df=df.drop(['pickup_datetime'],axis=1)
    
    return df

def clean_data(df):
    df = df.dropna(how = 'any', axis = 'rows')   
    # remove all not valid fare amounts and also extremely long examples to supress noise

    predicate = (df.fare_amount > 0) \
        & (df.distance < max_distance) \
        & (df.pickup_longitude <= + max_deviation) \
        & (df.pickup_longitude >= - max_deviation) \
        & (df.pickup_latitude <= + max_deviation) \
        & (df.pickup_latitude >= - max_deviation) \
        & (df.dropoff_longitude <= + max_deviation) \
        & (df.dropoff_longitude >= - max_deviation) \
        & (df.dropoff_latitude <= + max_deviation) \
        & (df.dropoff_latitude >= - max_deviation)
    
    df = df[predicate]    
    
    return df

#execute functions of train dataset
train = prepare_features(train_raw)
train = clean_data(train)

#split train data into input x and output y
y_train=train['fare_amount'].values
x_train=train.drop('fare_amount',axis=1).values

train.describe()
#define our model

activation = tf.nn.elu

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation=activation),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation=activation),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation=activation),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation=activation),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation=None),
])

def exp_decay(lr=0.011, k=0.1):
    def inner(epoch):
        return lr * exp(-k*epoch)
    
    return inner

#  some callbacks
best_callback   = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/best.ckpt', monitor='val_mse', save_best_only=True, verbose=1)
period_callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/{epoch:04d}.ckpt', monitor='val_mse', period=5, verbose=1)
scheduler       = tf.keras.callbacks.LearningRateScheduler(exp_decay())

# and our optimizer
opt = tf.keras.optimizers.Adam()

# define model loss and compile it
model.compile(optimizer=opt,
              loss='mse',
              metrics=['mae', 'mse'])

# finally let's train our model
metrics=model.fit(x_train, y_train,validation_split=0.1, epochs=25, batch_size=1024, callbacks=[scheduler, best_callback, period_callback])
definef, ax = plt.subplots(1,2,figsize = [15,5])

ax[0].plot(metrics.history['loss'])
ax[1].plot(metrics.history['mse'])
ax[1].plot(metrics.history['val_mse'])

ax[0].set_title('loss')
ax[1].set_title('performance')
ax[0].set_xlabel('epoch')
ax[1].set_xlabel('epoch')
ax[1].legend(['mse','val_mse'], loc='upper right')
plt.show()
best_model = tf.keras.models.load_model('/kaggle/working/best.ckpt')

test=pd.read_csv("/kaggle/input/test.csv",index_col=0)
x_test = prepare_features(test).values

y_test_predictions=best_model.predict(x_test)
submission = pd.DataFrame(
    {'key': test.index.values, 'fare_amount': y_test_predictions.squeeze()},
    columns = ['key', 'fare_amount'])
submission.to_csv('/kaggle/working/submission.csv', index = False)