# No need for presentations I guess
import numpy as np
import pandas as pd

# I import keras from the tensorflow library
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# Preprocessing and evaluation metric
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
def haversine(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        R = 6371

        lat = lat2 - lat1
        lng = lng2 - lng1

        d = np.sin(lat * 0.5)**2 \
            + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5)**2
        h = 2 * R * np.arcsin(np.sqrt(d))

        return h
def data_prep(raw, pred_data=False):
    out = raw.copy()
    # drop the 'id' column
    out.drop('id', axis=1, inplace=True)
    
    # add the haversine distance
    out.loc[:, 'distance'] = haversine(out.loc[:, 'pickup_latitude'],  \
                                       out.loc[:, 'pickup_longitude'], \
                                       out.loc[:, 'dropoff_latitude'], \
                                       out.loc[:, 'dropoff_longitude'])
    
    # split date_times
    elts = ['month', 'day', 'hour', 'minute', 'second']
    
    col = 'pickup_datetime'
    out[col] = pd.to_datetime(out[col])

    for el in elts:
        out[col + '_' + el] = out[col].map(lambda x: getattr(x, el))

    out[col + '_day_of_week'] = out[col].map(lambda x: x.isoweekday())
    
    # remove the original datetime column    
    out.drop('pickup_datetime', axis=1, inplace=True)
    
    # one hot encode categoricals :
    out = pd.get_dummies(out, columns=['vendor_id', 
                                       'store_and_fwd_flag', 
                                       'pickup_datetime_day_of_week'])
    
    # remove some outliers : trip longer than 22 hours and avg speed > 100km/h
    if not pred_data:
        out = out[out['trip_duration'] < 22 * 3600]
        out = out[out['distance'] / out['trip_duration'] * 3600 < 100 ]
        out.drop('dropoff_datetime', axis=1, inplace=True)
        
    # split the gps locations out of `out` 
    coords = ['pickup_latitude', 'pickup_longitude', 
              'dropoff_latitude' ,'dropoff_longitude']
    
    out_gps = out.loc[:, coords]
    out.drop(coords, axis=1, inplace=True)

    return pd.concat([out_gps, out], axis=1)
train = pd.read_csv('../input/train.csv')
train = data_prep(train)  

X = train.drop('trip_duration', axis=1)
y = train['trip_duration']

train = None
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_flat = X[:, 4:]
X_coords0 = X[:, :2]
X_coords1 = X[:, 2:4]
tb_path = 'tbGraphs/taxi/mult_shared/'
np.random.seed(1)
sub_n = 3 # number of layers in the shared subnet
lvl_n = 4 # number of layers for the second part of the network
n_node = 200 # number of neuron for each layer
coord_mod = Sequential()

coord_mod.add(Dense(n_node, activation='relu', input_dim=2))
for _ in range(sub_n - 1):
    coord_mod.add(Dense(n_node, activation='relu'))
coord_inputs0 = Input(shape=(2, ))
coord_inputs1 = Input(shape=(2, ))
shared_coord0 = coord_mod(coord_inputs0)
shared_coord1 = coord_mod(coord_inputs1)
merged_coord = keras.layers.concatenate([shared_coord0, shared_coord1])
flat_inputs = Input(shape=(X_flat.shape[1], ))
l = keras.layers.concatenate([merged_coord, flat_inputs])
for lnl in range(lvl_n):
    l = Dense(n_node, activation='relu')(l)
main_output = Dense(1, activation='relu', name='main_output')(l) 
model = Model(inputs=[coord_inputs0, coord_inputs1, flat_inputs], 
              outputs=main_output)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['msle', 'mse'])
batches = 512

path_grph = tb_path + 'shared/test' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node)
tbCallBack = keras.callbacks.TensorBoard(log_dir=path_grph,
                                         histogram_freq=0, 
                                         write_graph=True, 
                                         write_images=False)


path_mdl = 'shared_model' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node) + '.hdf5'
ckCallBack = keras.callbacks.ModelCheckpoint(path_mdl, 
                                             monitor='val_mean_squared_logarithmic_error',
                                             save_best_only=True,
                                             mode='min')
model.fit([X_coords0, X_coords1, X_flat], y, 
          batch_size=batches,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          callbacks=[tbCallBack, ckCallBack])
model.load_weights(path_mdl)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['msle', 'mse'])
path_grph = tb_path + 'shared/test_msle' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node)
tbCallBack = keras.callbacks.TensorBoard(log_dir=path_grph,
                                         histogram_freq=0, 
                                         write_graph=True, 
                                         write_images=False)
model.fit([X_coords0, X_coords1, X_flat], y, 
          batch_size=batches,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          callbacks=[tbCallBack, ckCallBack])
validation = pd.read_csv('../input/test.csv')
validation = data_prep(validation, pred_data=True)  
validation = scaler.transform(validation)

X_flat_val = validation[:, 4:]
X_coords_val0 = validation[:, :2]
X_coords_val1 = validation[:, 2:4]
submission = pd.read_csv('../input/sample_submission.csv')
model.load_weights(path_mdl)
keras_preds = model.predict([X_coords_val0, X_coords_val1, X_flat_val])
keras_submission = submission.copy()
keras_submission.trip_duration = keras_preds

keras_submission.to_csv('keras_submission.csv', index=False)