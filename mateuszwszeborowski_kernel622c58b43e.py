import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend
import numpy as np


def init_session():
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(0)
    sess=tf.compat.v1.Session(config=tf_config)

    tf.compat.v1.keras.backend.set_session(sess)

init_session()
train_raw=pd.read_pickle("/kaggle/input/etipgdla2020c1/train.pkl")
train_raw.head(10)

train_raw.describe()
import holidays
import math

def add_holiday_info(df):
    us_holidays = holidays.UnitedStates()  
    df['is_holiday'] = df.apply(lambda row: 1 if (row.pickup_datetime in us_holidays) else 0, axis=1)
    return df

def add_is_special_tarif_info(df):
    us_holidays = holidays.UnitedStates()
    df['is_special_tarif'] = df.apply(lambda row: 1 if ((row.day_of_week == 5 or row.day_of_week == 6) or (row.hour <= 6 or row.hour >= 20)) else 0, axis=1)
    return df

def get_distance(lat1, lon1, lat2, lon2):
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def add_distance(df):
    df['distance'] = df.apply(lambda row: get_distance(row.pickup_latitude, row.pickup_longitude, row.dropoff_latitude, row.dropoff_longitude), axis=1)
    jfk_coordinates = (40.6413111, -73.7781391);
    lga_coordinates = (40.7769271, -73.8739659);
    ewr_coordinates = (40.6895314, -74.1744623);
    
    df['jfk_pickup_distance'] = df.apply(lambda row: get_distance(row.pickup_latitude, row.pickup_longitude, jfk_coordinates[0], jfk_coordinates[1]), axis=1)
    df['lga_pickup_distance'] = df.apply(lambda row: get_distance(row.pickup_latitude, row.pickup_longitude, lga_coordinates[0], lga_coordinates[1]), axis=1)
    df['ewr_pickup_distance'] = df.apply(lambda row: get_distance(row.pickup_latitude, row.pickup_longitude, ewr_coordinates[0], ewr_coordinates[1]), axis=1)
    df['jfk_dropoff_distance'] = df.apply(lambda row: get_distance(row.dropoff_latitude, row.dropoff_longitude, jfk_coordinates[0], jfk_coordinates[1]), axis=1)
    df['lga_dropoff_distance'] = df.apply(lambda row: get_distance(row.dropoff_latitude, row.dropoff_longitude, lga_coordinates[0], lga_coordinates[1]), axis=1)
    df['ewr_dropoff_distance'] = df.apply(lambda row: get_distance(row.dropoff_latitude, row.dropoff_longitude, ewr_coordinates[0], ewr_coordinates[1]), axis=1)
    return df

def prepare_features(df):
    df = add_holiday_info(df)
    df = add_distance(df)
    dtcol=pd.DatetimeIndex(df['pickup_datetime'])    
    df['year']=dtcol.year
    df['hour']=dtcol.hour
    df['day_of_week']=dtcol.dayofweek
    df['day'] = dtcol.day
    df['month'] =  dtcol.month
    df['quarter_of_year'] =  dtcol.quarter
    df = add_is_special_tarif_info(df)

    df['diff_longitude'] = (df.dropoff_longitude -df.pickup_longitude).abs()
    df['diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['distance_euclidean'] = (df['diff_latitude'] ** 2 + df['diff_longitude'] ** 2) ** 0.5

    #since we designed more valuable feature we will not need original timestamp
    df=df.drop(['pickup_datetime'],axis=1)
    return df

def clean_data_before_features(df):
    df = df.dropna(how = 'any', axis = 'rows')   
    df = df[(df.fare_amount>0) & (df.fare_amount < 300)]
    df = df[(df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42) & (df.pickup_latitude >= 38) & (df.pickup_latitude <= 42)]
    df = df[(df.pickup_longitude >= -76) & (df.pickup_longitude <= -70) & (df.pickup_longitude >= -76) & (df.pickup_longitude <= -70)]
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    return df

def clean_data_after_features(df):
    df = df.drop('passenger_count', 1)
    return df

#execute functions of train dataset
train = clean_data_before_features(train_raw)
train = prepare_features(train)
train = clean_data_after_features(train)

#split train data into input x and output y
y_train=train['fare_amount'].values
x_train=train.drop('fare_amount',axis=1)
x_train=train.drop('fare_amount',axis=1)

mean = x_train.mean(axis = 0)

x_train -= mean

std = x_train.std(axis = 0)

x_train /= std

x_train = x_train.values
from keras.layers import BatchNormalization

#define our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation=tf.nn.selu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation=None)
])

#and our optimizer
opt = keras.optimizers.Adam(learning_rate =0.001, beta_1=0.9, beta_2=0.999)

#define model loss and compile it
model.compile(optimizer=opt,
              loss='mse',
              metrics=['mse'])

#finally let's train our model
metrics=model.fit(tf.convert_to_tensor(x_train, np.float32), tf.convert_to_tensor(y_train, np.float32),validation_split=0.01, shuffle= True, epochs=30, verbose=1, batch_size=256)
!pip install -U keras-tuner
# DISCLAIMER - this code was not used in a final result due to the fact described above
'''from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

class TaxiTariffRegressionHyperModel(HyperModel):
    def build(self, hp):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation=tf.nn.selu),
            tf.keras.layers.Dropout( 
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                )),
            tf.keras.layers.Dense(12, activation=tf.nn.selu),
            tf.keras.layers.Dropout( 
                rate=hp.Float(
                    "dropout_2", min_value=0.0, max_value=0.2, default=0.1, step=0.05,
            )),
            tf.keras.layers.Dense(
                units=hp.Int(
                    "units", min_value=4, max_value=8, step=1, default=6
                ),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "selu"],
                    default="selu",
                ),),
            tf.keras.layers.Dense(1, activation=None)
        ])

        opt = tf.keras.optimizers.Adam(
            hp.Float(
                "learning_rate",
                min_value=1e-4,
                max_value=1e-2,
                sampling="LOG",
                default=1e-3,
            ),
            beta_1=0.9, beta_2=0.999, amsgrad=False,
        )

        model.compile(optimizer=opt,
                      loss='mse',
                      metrics=['mse'])
        
        return model


hypermodel = TaxiTariffRegressionHyperModel()
tuner = Hyperband(
        hypermodel,
        objective="mse",
        seed=1,
        max_epochs=5,
        executions_per_trial=2,
    )

tuner.search(tf.convert_to_tensor(x_train, np.float32), tf.convert_to_tensor(y_train, np.float32), epochs=5, validation_split=0.01)
model = tuner.get_best_models(num_models=1)[0]
'''
test=pd.read_csv("/kaggle/input/etipgdla2020c1/test.csv",index_col=0)
x_test = prepare_features(test)
x_test = clean_data_after_features(x_test)

# Scale the test set accordingly (Z-Score)
x_test -= mean
x_test /= std

y_test_predictions=model.predict(x_test)
submission = pd.DataFrame(
    {'key': test.index.values, 'fare_amount': y_test_predictions.squeeze()},
    columns = ['key', 'fare_amount'])
submission.to_csv('submissionMinMax11.csv', index = False)