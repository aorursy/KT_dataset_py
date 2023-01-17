import os
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from keras.layers import Input, Dense, Bidirectional, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping

import numpy as np
os.listdir("../input/")
coin_dataframes = {}

def convert_comma_int(field):
    try:
        return int(field.replace(',', ''))
    except ValueError:
        return None
    
for fn in os.listdir("../input/"):
    if "bitcoin_cache" in fn:
        continue
    if fn.endswith("_price.csv"):
        coin_name = fn.split("_")[0]
        df = pd.read_csv(os.path.join("../input/", fn), parse_dates=["Date"])
        df['Market Cap'] = df['Market Cap'].map(convert_comma_int)
        coin_dataframes[coin_name] = df.sort_values('Date')
coin_dataframes.keys()
coin_dataframes['nem'].head()
# import plotly.plotly as py
# import plotly.graph_objs as go

# from datetime import datetime
# # import pandas_datareader.data as web

# data = [go.Scatter(x=coin_dataframes['bitcoin'].Date, y=coin_dataframes['bitcoin'].Date)]

# py.iplot(data)

plt.figure(figsize=(20,8))
coin_dataframes['bitcoin'].plot(x='Date', y='Close')
plt.show()
def add_relative_columns(df):
    day_diff = df['Close'] - df['Open']
    df['rel_close'] = day_diff / df['Open']
    df['high_low_ratio'] = df['High'] / df['Low']
    df['rel_high'] = df['High'] / df['Close']
    df['rel_low'] = df['Low'] / df['Close']
    
    
for df in coin_dataframes.values():
    add_relative_columns(df)
    
coin_dataframes["nem"].head()
def create_history_frames(coin_dataframes):
    history_frames = {}
    for coin_name, df in coin_dataframes.items():
        history_frames[coin_name], x_cols = create_history_frame(df)
    return history_frames, x_cols
        

def create_history_frame(df):
    feature_cols = ['rel_close', 'rel_high', 'rel_low', 'high_low_ratio']
    y_col = ['rel_close']
    x_cols = []
    days = 10
    history = df[['Date'] + y_col].copy()
    for n in range(1, days+1):
        for feat_col in feature_cols:
            colname = '{}_{}'.format(feat_col, n)
            history[colname] = df[feat_col].shift(n)
            x_cols.append(colname)
    history = history[days:]
    return history, x_cols

y_col = 'rel_close'
coin_history, x_cols = create_history_frames(coin_dataframes)
def create_model():
    input_layer = Input(batch_shape=(None, len(x_cols), 1))
    layer = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    layer = Bidirectional(LSTM(128))(layer)
    out = Dense(1, activation="sigmoid")(layer)
    m = Model(inputs=input_layer, outputs=out)
    m.compile("rmsprop", loss='mean_squared_error')
    return m

def create_train_test_mtx(history):
    X = history[x_cols].as_matrix()
    y = history[y_col].as_matrix()
    X = X.reshape(X.shape[0], X.shape[1], 1)
    rand_mtx = np.random.permutation(X.shape[0])
    train_split = int(X.shape[0] * 0.9)
    train_indices = rand_mtx[:train_split]
    test_indices = rand_mtx[train_split:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def train_model(model, X, y):
    ea = EarlyStopping(monitor='val_loss', patience=2)
    val_loss = model.fit(X, y, epochs=500, batch_size=64, callbacks=[ea], verbose=1, validation_split=.1)
    return val_loss
rmse = {}
pred = {}
test = {}

for coin_name, history in coin_history.items():
    model = create_model()
    X_train, X_test, y_train, y_test = create_train_test_mtx(history)
    train_model(model, X_train, y_train)
    test[coin_name] = y_test
    
    # run prediction on test set
    pred[coin_name] = model.predict(X_test)
    # compute test loss
    rmse[coin_name] = np.sqrt(np.mean((pred[coin_name] - y_test)**2))
    print(coin_name, rmse[coin_name])
pred_sign = {coin_name: np.sign(pred[coin_name]) * np.sign(test[coin_name]) for coin_name in pred.keys()}
for coin, val in sorted(pred_sign.items()):
    cnt = np.unique(pred_sign[coin], return_counts=True)[1]
    print("[{}] pos/neg change guessed correctly: {}, incorrectly: {}, correct%: {}".format(
        coin, cnt[0], cnt[1], cnt[0]/ (cnt[0]+cnt[1]) * 100))
pred_sign = {coin_name: np.sign(pred[coin_name]) for coin_name in pred.keys()}
for coin, val in sorted(pred_sign.items()):
    e, cnt = np.unique(val, return_counts=True)
    print("[{}] guesses: {}".format(coin, dict(zip(e, cnt))))