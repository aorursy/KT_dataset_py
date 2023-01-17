import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense
RANDOM_SEED=5
train = pd.read_csv('../input/idao2020/data/train.csv', index_col=0)

test =  pd.read_csv('../input/idao2020/data/Track 1/test.csv', index_col=0)
def prepare_features(df):

    '''minimal preprocessing'''

    date = pd.to_datetime(df.epoch)

    # year and month are the same accross the data

    df['day'] = date.dt.day

    df['weekday'] = date.dt.weekday

    df['hour'] = date.dt.hour

    df['minute'] = date.dt.minute

    df['second'] = date.dt.second

    

    return df.drop('epoch', axis=1)
train = prepare_features(train)

X = train[['x_sim', 'y_sim', 'z_sim',

           'Vx_sim', 'Vy_sim', 'Vz_sim',

           'sat_id', 'day', 'weekday', 'hour', 'minute','second']]

Y = train[['x', 'y', 'z',

           'Vx', 'Vy', 'Vz']]
from sklearn.model_selection import KFold
train_idx, test_idx = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, Y['x'], groups=X['sat_id']))[0]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]



# Normalize features

norm_cols = ['x_sim', 'y_sim', 'z_sim',

           'Vx_sim', 'Vy_sim', 'Vz_sim']

X_train[norm_cols] = (X_train[norm_cols] - X_train[norm_cols].mean(axis=0)) / X_train[norm_cols].std(axis=0)

X_test[norm_cols] = (X_test[norm_cols] - X_test[norm_cols].mean(axis=0)) / X_test[norm_cols].std(axis=0)



y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)

y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)
# X_train = X_train.values

# y_train = y_train.values
# Build the model.

model = Sequential([

  Dense(12, activation='relu', input_shape=(12,)),

  Dense(48, activation='relu'),

  Dense(24, activation='relu'),

  Dense(6, activation='linear'),

])



# Compile the model.

model.compile(

  optimizer='adam',

  loss='mean_squared_error',

  metrics=['mean_squared_error'],

)
# Train the model.

model.fit(

    X_train,

    y_train,

    epochs=5,

    batch_size=25000,

)
y_pred = model.predict(X_test)
y_pred
y_test