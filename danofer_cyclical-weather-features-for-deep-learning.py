import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')
cities_temperature = pd.read_csv("../input/temperature.csv", parse_dates=['datetime'])
cities_temperature.sample(5)
data = cities_temperature[['datetime', 'Montreal']]
data = data.rename(columns={'Montreal': 'temperature'})
data['temperature'] = data['temperature'] - 273.15
print(data.dtypes)
data.head(5)
data = data.fillna(method = 'bfill', axis=0).dropna()
print(data.temperature.describe())
ax = sns.distplot(data.temperature)
data['hour'] = data.datetime.dt.hour
sample = data[:168] # roughly the first week of the data
ax = sample['hour'].plot()
sample[9:14]
data['hour_sin'] = np.sin(2 * np.pi * data['hour']/23.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/23.0)
sample = data[0:168]
ax = sample['hour_sin'].plot()
sample[10:26]
ax = sample.plot.scatter('hour_sin', 'hour_cos').set_aspect('equal')
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
data['month'] = data.datetime.dt.month
data = encode(data, 'month', 12)

data['day'] = data.datetime.dt.month
data = encode(data, 'day', 365)
data.head()
data.to_csv("montreal_hourlyWeather_cyclicEncoded.csv.gz",index=False,compression="gzip")
from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data, test_size=0.4)
data_test, data_val = train_test_split(data_test, test_size=0.5)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam

def train_model(X_train, y_train, X_test, y_test, epochs):
    model = Sequential(
        [
            Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
#             Dense(10, activation="relu"),
#             Dense(10, activation="relu"),
            Dense(1, activation="linear")
        ]
    )
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model, history
def get_unencoded_features(df):
    return df[['month', 'day', 'hour']]
X_train = get_unencoded_features(data_train)
X_test  = get_unencoded_features(data_test)
y_train = data_train.temperature
y_test  = data_test.temperature
model_unencoded, unencoded_hist = train_model(
    get_unencoded_features(data_train),
    data_train.temperature,
    get_unencoded_features(data_test),
    data_test.temperature,
    epochs=5
)
def get_encoded_features(df):
    return df[['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']]
X_train = get_encoded_features(data_train)
X_test  = get_encoded_features(data_test)
y_train = data_train.temperature
y_test  = data_test.temperature
model_encoded, encoded_hist = train_model(
    get_encoded_features(data_train),
    data_train.temperature,
    get_encoded_features(data_test),
    data_test.temperature,
    epochs=5
)
plt.plot(unencoded_hist.history['val_loss'], "r")
ax = plt.plot(encoded_hist.history['val_loss'], "b")
X_val_unencoded  = get_unencoded_features(data_val)
X_val_encoded  = get_encoded_features(data_val)
y_val = data_val.temperature
from sklearn.metrics import mean_squared_error
mse_unencoded = mean_squared_error(y_val, model_unencoded.predict(X_val_unencoded))
print(mse_unencoded)
mse_encoded = mean_squared_error(y_val, model_encoded.predict(X_val_encoded))
print(mse_encoded)
print('We achieved an improvement of {0:.2f}% in our MSE'.format((mse_unencoded - mse_encoded)/mse_unencoded * 100))
