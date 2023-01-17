import numpy as np

import pandas as pd

from datetime import datetime

from matplotlib import pyplot as plt

from math import sqrt



from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.metrics import mean_squared_error



from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, LSTM, Dropout, CuDNNGRU, GRU, Bidirectional

from keras.optimizers import RMSprop, adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers.advanced_activations import LeakyReLU





import os

print(os.listdir("../input"))
def parse(x):

    return datetime.strptime(x, '%Y %m %d %H')
df = pd.read_csv("../input/beijing.csv", parse_dates = [['year', 'month', 'day', 'hour']], date_parser = parse, index_col=0)

df.head()
df.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

df.index.name = 'date'

df["pollution"].fillna(0, inplace=True)

df = df[24:]

df.head()
values = df.values

# specify columns to plot

groups = [0, 1, 2, 3, 5, 6, 7]

i = 1

plt.figure(figsize=(12, 12))

for group in groups:

    plt.subplot(len(groups), 1, i)

    plt.plot(values[:, group])

    plt.title(df.columns[group], y=0.5, loc='right')

    i += 1

plt.show()
df1 = df.copy()

df1.head()
encoder = LabelEncoder()

df1["wnd_dir"] = encoder.fit_transform(df1["wnd_dir"])



df1 = df1.astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))

df_scaled = scaler.fit_transform(df1)



df_scaled = pd.DataFrame(df_scaled, index=df1.index, columns=df1.columns)

df_scaled.head()
def series_to_supervised(data, output, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = [], []

    

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        for j in df.columns:

            names.append("{}(t-{})".format(j, i))

            

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df[output].shift(-i))

        if i == 0:

            names.append("{}(t)".format(output))

        else:

            names.append("{}(t + {})".format(output, i))

            

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    

    if dropnan:

        agg.dropna(inplace=True)

    return agg
output = "pollution"

input_lag = 5

reframed = series_to_supervised(df_scaled, output, input_lag, 1)

reframed.shape
n_train_hours = 365 * 24 * 4



values = reframed.values

train = values[:n_train_hours, :]

test = values[n_train_hours:, :]



train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], input_lag, int(train_X.shape[1] / input_lag)))

test_X = test_X.reshape((test_X.shape[0], input_lag, int(test_X.shape[1] / input_lag)))



print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# define model

n_steps, n_features = input_lag, 8

model = Sequential()

model.add(Bidirectional(CuDNNGRU(128, input_shape=(n_steps, n_features), return_sequences=True, name = "GRU_1")))

model.add(LeakyReLU(alpha=0.01, name = "LreLU_1"))

model.add(BatchNormalization(name = "BN_1"))



model.add(Bidirectional(CuDNNGRU(256, name = "GRU_2")))

model.add(LeakyReLU(alpha=0.01, name = "LreLU_2"))

model.add(BatchNormalization(name = "BN_2"))

model.add(Dropout(0.1, name = "DROP_1"))



model.add(Dense(1, name = "Dense_1"))

optimizer = RMSprop(lr = 0.001, epsilon = 1e-08, decay = 0.0)

model.compile(optimizer=optimizer, loss='mse')

#model.summary()
callbacks = [

    EarlyStopping(

        monitor = 'val_loss', 

        patience = 10,

        mode = 'min',

        verbose = 1),

    ReduceLROnPlateau(

        monitor = 'val_loss', 

        patience = 3, 

        verbose = 1, 

        factor = 0.5, 

        min_lr = 0.00001)]
history = model.fit(train_X, train_y, 

                    epochs=100, 

                    batch_size=256, 

                    validation_data=(test_X, test_y), 

                    verbose=2, 

                    shuffle=False,

                    callbacks = callbacks)
plt.plot(history.history['loss'], color = 'b', label = "Training loss")

plt.plot(history.history['val_loss'], color = 'r', label = "Test loss")

plt.legend()

plt.show()
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], input_lag*8))



# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, -7:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X[:, -7:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]
sqrt(mean_squared_error(inv_y, inv_yhat))
plt.figure(figsize=(20, 4))

plt.plot(inv_y[:100000], "r")

plt.plot(inv_yhat[:100000], "b")
max(abs(inv_y - inv_yhat))
plt.plot(inv_y - inv_yhat)
from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model.png")