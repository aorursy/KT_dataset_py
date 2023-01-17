import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sns.set()
%matplotlib inline
with pd.HDFStore('../input/air-quality-madrid/madrid.h5') as data:
    df = data['28079016']
    
df = df.sort_index()
df.info()
msno.matrix(df, freq='Y')
fig, ax = plt.subplots(figsize=(20, 5))

candidates = df[['CO', 'NO_2', 'O_3']] 

candidates /= candidates.max(axis=0)

(candidates.interpolate(method='time')
           .rolling(window=24*15).mean()
           .plot(ax=ax))
def pivot_with_offset(series, offset):
    pivot = pd.DataFrame(index=df.index)

    for t in range(offset * 2):
        pivot['t_{}'.format(t)] = series.shift(-t)

    pivot = pivot.dropna(how='any')
    return pivot


offset = 24

series = (df.NO_2.interpolate(method='time')
                 .pipe(pivot_with_offset, offset)
                 .apply(np.log, axis=1)
                 .replace(-np.inf))

# Get only timestamps at 00:00 and 12:00
series = series[(series.index.hour % 12) == 0]

# Make it a multiple of the chosen batch_size
if series.shape[0] % 32 != 0:
    series = series.iloc[:-(series.shape[0]%32)]
test_ratio = 0.2

split_point = int(series.shape[0] * (1 - test_ratio))
split_point -= split_point % 32

np_series = series.values

X_train = series.values[:split_point , :offset]
y_train = series.values[:split_point, offset:]
X_test = series.values[split_point:, :offset]
y_test = series.values[split_point:, offset:]
# Scale only to train data to prevent look-ahead bias
lift = X_train.min()
scale = X_train.max()

def scale_array(arr, lift, scale):
    return (arr - lift) / scale

X_train = np.expand_dims(scale_array(X_train, lift, scale), axis=2)
y_train = np.expand_dims(scale_array(y_train, lift, scale), axis=2)
X_test = np.expand_dims(scale_array(X_test, lift, scale), axis=2)
y_test = np.expand_dims(scale_array(y_test, lift, scale), axis=2)
def create_lstm(offset, neurons=(2,1), batch_size=32, lr=0.005, n_features=1):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        neurons[0], return_sequences=True, stateful=True, 
        batch_input_shape=(batch_size, offset, n_features))
    )
    
    
    # Second LSTM layer if defined
    if neurons[1]:
        model.add(LSTM(
            neurons[1], return_sequences=True, stateful=True, 
            batch_input_shape=(batch_size, offset, n_features))
        )
    
    # TimeDistributed layer to generate all the timesteps
    model.add(TimeDistributed(Dense(1)))
    
    optimizer = RMSprop(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model
def train_model(model, X_train, y_train, batch_size=32, epochs=20):
    mse = list()

    for i in range(epochs):
        if i % 1 == 0:
            print('Epoch {:02d}/{}...'.format(i + 1, epochs), end=' ')

        log = model.fit(
            X_train, y_train, 
            epochs=1, batch_size=32, 
            verbose=0, shuffle=False
        )
    
        mse.append(log.history['loss'][-1])
        print('loss: {:.4f}'.format(mse[-1]))
    
        model.reset_states()
        
    return model, mse
def validate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    model.reset_states()
    return np.mean((y_test - preds) ** 2)
model = create_lstm(offset)
model, _ = train_model(model, X_train, y_train)

model.fit(
    X_train, y_train, 
    epochs=1, batch_size=32, 
    verbose=1, shuffle=False
)
model.reset_states()

preds = model.predict(X_test)
'MSE: {:.5f}'.format(validate_model(model, X_test, y_test))
fig, ax = plt.subplots(figsize=(20, 5))

start = X_test.shape[0] - 42
interval = X_test[start+3:start+33:2, :, 0].reshape(-1, 1)

truth, = plt.plot(np.arange(24*15), interval, alpha=0.6)

old_preds = list()

for point in range(1, 15, 2):
    prediction = np.squeeze(preds[start + point*2])
    pred, = plt.plot(point * offset + np.arange(offset) - 12, prediction,
                     linestyle='--', color='red')
    old_preds.append(prediction)

plt.legend(
    [truth, pred],
    ['Observation', 'Prediction']
)
ax.set_ylim([-.1, 1.1])
ax.set_xticks(12 + np.arange(15) * offset)
_ = ax.set_xticklabels([])
traffic = (pd.read_csv('../input/traffic-in-madrid/traffic.csv', parse_dates=['date'])
             .rename({'intensidad': 'traffic'}, axis=1)
             .set_index('date'))

_ = (traffic.rolling(window=24*7).mean()
            .plot(figsize=(20,5)))
df = (df.reset_index()
        .merge(traffic.reset_index(), on='date')
        .set_index('date'))
df['weekend'] = ((df.index.weekday == 5) | (df.index.weekday == 6)).astype(np.int32)
air = (df.NO_2.interpolate(method='time')
              .pipe(pivot_with_offset, offset)
              .apply(np.log, axis=1)
              .replace(-np.inf))

traf = pivot_with_offset(df.traffic, offset)
week = pivot_with_offset(df.weekend, offset)

# Get only timestamps at 00:00 and 12:00
air = air[(air.index.hour % 12) == 0]
traf = traf[(traf.index.hour % 12) == 0]
week = week[(week.index.hour % 12) == 0]

# Substitute nans with zeros
air = air.fillna(0)
traf = traf.fillna(0)
week = week.fillna(0)

all_data = np.dstack([air.values, traf.values, week.values, week.shift(-1)])

if all_data.shape[0] % 32 != 0:
    all_data = all_data[:-(all_data.shape[0]%32)]
    
all_data.shape
test_ratio = 0.2

split_point = int(all_data.shape[0] * (1 - test_ratio))
split_point -= split_point % 32

X_train = all_data[:split_point , :offset, :]
y_train = all_data[:split_point, offset:, :1]
X_test = all_data[split_point:, :offset, :]
y_test = all_data[split_point:, offset:, :1]
X_train[:,:,0] = scale_array(X_train[:,:,0], lift, scale)
y_train[:,:,0] = scale_array(y_train[:,:,0], lift, scale)
X_test[:,:,0] = scale_array(X_test[:,:,0], lift, scale)
y_test[:,:,0] = scale_array(y_test[:,:,0], lift, scale)

# Scale traffic differently to prevent errors by assuming columns
lift = X_train[:,:,1].min()
scale = X_train[:,:,1].max()

X_train[:,:,1] = scale_array(X_train[:,:,1], lift, scale)
X_test[:,:,1] = scale_array(X_test[:,:,1], lift, scale)
model = create_lstm(offset, lr=0.025, n_features=4)
model, _ = train_model(model, X_train, y_train, epochs=50)

model.fit(
    X_train, y_train, 
    epochs=1, batch_size=32, 
    verbose=1, shuffle=False
)

preds = model.predict(X_test)
model.reset_states()
'MSE: {:.5f}'.format(validate_model(model, X_test, y_test))
fig, ax = plt.subplots(figsize=(20, 5))

start = X_test.shape[0] - 42
interval = X_test[start+3:start+33:2, :, :]

truth, = plt.plot(np.arange(24*15), interval[:, :, 0].reshape(-1, 1), alpha=0.6)

for point, old_pred in zip(range(2, 15, 2), old_preds):
    prediction = np.squeeze(preds[start + point*2])
    old, = plt.plot(point * offset + np.arange(offset) - 12, old_pred,
                    linestyle='--', color='red', alpha=0.4)
    new, = plt.plot(point * offset + np.arange(offset) - 12, prediction,
                    linestyle='--', color='green')

plt.legend(
    [truth, old, new],
    ['Observation', 'Original prediction', 'New model']
)
    
ax.set_ylim([-.1, 1.1])
ax.set_xticks(12 + np.arange(15) * offset)
_ = ax.set_xticklabels([])