import pandas as pd
df_train = (pd.read_csv('../input/train.csv',nrows = 10000000).drop(['key'],axis=1))
df_test = pd.read_csv('../input/test.csv')
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split

df_train, df_validation = train_test_split(df_train, test_size=0.10, random_state=1)

validation_target = df_validation['fare_amount'].values

def clean_dataframe(df):
    
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]

   # df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]

    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    df = df[(1 <= df['passenger_count']) & (df['passenger_count'] <= 6)]
    
    return df
  
    
def distance(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)   
    
def add_features(df):
    plat = df['pickup_latitude']
    dlat = df['dropoff_latitude']
    plon = df['pickup_longitude']
    dlon = df['dropoff_longitude']


    df['latitude_diff'] = (plat - dlat)
    df['longitude_diff'] = (plon - dlon)
    
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['quarter_of_year'] = df['pickup_datetime'].dt.quarter
    
    lgr = (-73.8733, 40.7746)
    jfk = (-73.7900, 40.6437)
    ewr = (-74.1843, 40.6924)

    df['trip_distance_km'] = distance(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)   
    df['pickup_distance_jfk'] = distance(df['pickup_latitude'], df['pickup_longitude'], jfk[1], jfk[0])
    df['dropoff_distance_jfk'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], jfk[1], jfk[0])
    df['pickup_distance_ewr'] = distance(df['pickup_latitude'], df['pickup_longitude'], ewr[1], ewr[0])
    df['dropoff_distance_ewr'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], ewr[1], ewr[0])
    df['pickup_distance_laguardia'] = distance(df['pickup_latitude'], df['pickup_longitude'], lgr[1], lgr[0])
    df['dropoff_distance_laguardia'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], lgr[1], lgr[0]) 
    
    return df


df_train = clean_dataframe(df_train)
df_validation = clean_dataframe(df_validation)
df_train = add_features(df_train)
df_validation = add_features(df_validation)
df_test = add_features(df_test)

import matplotlib.pyplot as plt
import seaborn as sns
colormap = 'Blues'
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)




print(df_train.head(5))
print(df_test.head(5))

print(df_train.corrwith(df_train['fare_amount']))


paratest = df_test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','latitude_diff','longitude_diff','trip_distance_km','pickup_distance_ewr','dropoff_distance_ewr','pickup_distance_laguardia','dropoff_distance_laguardia','year']]
para = df_train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','latitude_diff','longitude_diff','trip_distance_km','pickup_distance_ewr','dropoff_distance_ewr','pickup_distance_laguardia','dropoff_distance_laguardia','year']]
paraval = df_validation[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','latitude_diff','longitude_diff','trip_distance_km','pickup_distance_ewr','dropoff_distance_ewr','pickup_distance_laguardia','dropoff_distance_laguardia','year']]

target = df_train.fare_amount
validation_target = df_validation.fare_amount

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_train = scaler.fit_transform(para)
df_validation = scaler.transform(paraval)
df_test = scaler.transform(paratest)

n_col = para.shape[1]
print(n_col)

from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import keras.backend as K

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def model() :   #Created a Model using Keras
    model = Sequential()
    model.add(Dropout(0.2,input_shape=(n_col,)))
    model.add(BatchNormalization())
    model.add(Dense(512,activation='relu'))#512 neurons in input layer
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256,activation='relu')) #256 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))  # 128 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))   # 64 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))   # 32 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(16,activation='relu')) # 16 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(8,activation='relu')) # 8 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(1)) # 1 neuron in output layer
    
    #nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #adadelta =optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #adgrad = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    #rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.5)

    model.compile(optimizer='adam',loss='mse', metrics=[rmse])
    return model


model = model()
estimator = model.fit(x=para,y=target, batch_size=1024, epochs=1, 
                    verbose=1, validation_data=(paraval,validation_target), 
                    shuffle=True)
model.summary()
pred_y = model.predict(paratest)
df_final = pd.read_csv('../input/test.csv')
df_final['pred'] = pred_y
submission = pd.DataFrame(
    {'key': df_final.key, 'fare_amount': df_final.pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_DNN.csv', index = False)
submission.head(5)