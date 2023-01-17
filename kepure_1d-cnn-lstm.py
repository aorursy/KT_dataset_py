import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
x_train_data = pd.read_csv('../input/X_train.csv')

y_train_data = pd.read_csv('../input/y_train.csv')

x_test_data = pd.read_csv('../input/X_test.csv')
x_train_data.head()
x_train_data.drop(['series_id', 'measurement_number'], axis=1).describe()
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.degrees(math.atan2(t0, t1))



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.degrees(math.asin(t2))



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.degrees(math.atan2(t3, t4))



    return X, Y, Z
def feature_engineering(data):

    eulers = np.array([quaternion_to_euler(*x) for x in data[['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']].values])

    data['euler_orientation_x'] = eulers[:, 0]

    data['euler_orientation_y'] = eulers[:, 1]

    data['euler_orientation_z'] = eulers[:, 2]



    accCols = ['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

    gyroCols = ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']

    quatCols = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']

    gyroEulerCols = ['euler_orientation_x', 'euler_orientation_y', 'euler_orientation_z']



    data['angular_velocity_norm'] = np.sqrt(np.sum(np.square(data[gyroCols]),axis=1))

    data['orientation_norm'] = np.sqrt(np.sum(np.square(data[quatCols]),axis=1))

    data['linear_acceleration_norm'] = np.sqrt(np.sum(np.square(data[accCols]),axis=1))

    data['euler_orientation_norm'] = np.sqrt(np.sum(np.square(data[gyroEulerCols]),axis=1))

    data['acc_vs_vel'] = data['linear_acceleration_norm'] / data['angular_velocity_norm']

    

    return data



x_train_data = feature_engineering(x_train_data)

x_test_data = feature_engineering(x_test_data)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

original_series_id = x_train_data['series_id']

df_x_train = pd.DataFrame(sc.fit_transform(x_train_data.drop(['row_id', 'series_id', 'measurement_number'], axis=1)),

                       columns=x_train_data.drop(['row_id', 'series_id', 'measurement_number'], axis=1).columns)

df_x_train['series_id'] = original_series_id

df_x_train.describe()
original_series_id = x_test_data['series_id']

df_x_test = pd.DataFrame(sc.transform(x_test_data.drop(['row_id', 'series_id', 'measurement_number'], axis=1)),

                       columns=x_test_data.drop(['row_id', 'series_id', 'measurement_number'], axis=1).columns)

df_x_test['series_id'] = original_series_id
series_id = x_train_data.series_id.unique()

set([len(df_x_train[df_x_train['series_id'] == row_id]) for row_id in series_id])
def get_sliding_window(x, y, seq_length, shift):

    data_grouped_by_classes = {key: [] for key in set(y['surface'].values)}

    for class_name in set(y['surface'].values):

        for y_row in y.iterrows():

            y_row = y_row[1]

            if y_row['surface'] == class_name:

                data_grouped_by_classes[class_name].append([x[x['series_id'] == y_row['series_id']], y_row['surface']])       

    

    x_ret = []

    y_ret = []

    for class_name in data_grouped_by_classes.keys():

        for arr in data_grouped_by_classes[class_name]:

            arr_y = arr[1]

            arr_x = arr[0].values

            batch_size_total = seq_length

            n_batches = len(arr_x)//batch_size_total

            arr_x = arr_x[:n_batches * batch_size_total]

            for n in range(0, arr_x.shape[0], shift):  

                if n + seq_length > arr_x.shape[0]:

                    break

                x_batch = arr_x[n:n+seq_length]

                y_batch = arr_y

                x_ret.append(x_batch)

                y_ret.append(y_batch)

    return x_ret, y_ret



def get_data(x, y, batch_size, seq_length, shift):

    import random

    data_x = []

    data_y = []

    for x_, y_ in zip(*get_sliding_window(x, y, seq_length, shift)):

        data_x.append(x_)

        data_y.append(y_)

    

    c = list(zip(data_x, data_y))

    random.shuffle(c)

    data_x, data_y = zip(*c)

    ret_x = []

    ret_y = []

    

    for i in range(0, len(data_x), batch_size):

        if len(data_x) <= i+batch_size:

            break

        ret_x.append(np.array(data_x[i:i+batch_size])[0])

        ret_y.append(np.array(data_y[i:i+batch_size])[0])

    return np.array(ret_x), np.array(ret_y)
x_train, y_train = get_data(df_x_train, y_train_data, 1, 16, 2)

x_train = x_train[:, :, :-1]

x_train = x_train.reshape(-1, 4, 4, 18)
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



enc_l = LabelEncoder()

enc = OneHotEncoder(handle_unknown='ignore')

nr_y_train = enc_l.fit_transform(np.array(y_train))

one_hot_y_train = enc.fit_transform(nr_y_train.reshape(-1, 1))
import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, LSTM, GRU, SimpleRNN, Conv1D, TimeDistributed, MaxPooling1D, Flatten, Dropout

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import EarlyStopping
def init_model():

    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'), batch_input_shape=(None, None, 4, 18)))

    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(16))

    model.add(Dense(9, activation='softmax'))

    

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)



    model.compile(optimizer=adam,

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = init_model()

callbacks = [EarlyStopping('val_loss', patience=3)]

model.fit(x_train, one_hot_y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=callbacks)
series_id = df_x_test.series_id.unique()

our_predictions = []

for row_id in series_id:

    batch_data = df_x_test[df_x_test['series_id'] == row_id].drop(['series_id'], axis=1).values[:128, :].reshape(8, 16, 18).reshape(8, 4, 4, 18)

    predictions = model.predict(batch_data)

    final_predictions = list(sum(predictions)/len(predictions))

    our_predictions.append(final_predictions)



label_rez = enc_l.inverse_transform(enc.inverse_transform(our_predictions))



df = pd.DataFrame({'series_id': series_id.tolist(),'surface': label_rez.tolist()})

df.to_csv('submission.csv', index=False)