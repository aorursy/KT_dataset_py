import numpy as np

import tensorflow as tf

import pandas as pd

pd.options.mode.chained_assignment = None

import seaborn as sns

from matplotlib.pylab import rcParams

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline



sns.set(style='whitegrid', palette='muted')

rcParams['figure.figsize'] = 14, 8

np.random.seed(1)

tf.random.set_seed(1)



print('Tensorflow version:', tf.__version__)
df = pd.read_csv("../input/sp500-daily-19862018/spx.csv", parse_dates=['date'])

df.head()
df.shape
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.date, y=df.close,

                    mode='lines',

                    name='close'))

fig.update_layout(showlegend=True)

fig.show()
train_size = int(len(df) * 0.8)

test_size = len(df) - train_size

train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(train.shape, test.shape)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler = scaler.fit(train[['close']])



train['close'] = scaler.transform(train[['close']])

test['close'] = scaler.transform(test[['close']])
def create_dataset(X, y, time_steps=1):

    Xs, ys = [], []

    for i in range(len(X) - time_steps):

        v = X.iloc[i:(i + time_steps)].values

        Xs.append(v)        

        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)
time_steps = 30



X_train, y_train = create_dataset(train[['close']], train.close, time_steps)

X_test, y_test = create_dataset(test[['close']], test.close, time_steps)



print(X_train.shape)
timesteps = X_train.shape[1]

num_features = X_train.shape[2]
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed



model = Sequential([

    LSTM(128, input_shape=(timesteps, num_features)),

    Dropout(0.2),

    RepeatVector(timesteps),

    LSTM(128, return_sequences=True),

    Dropout(0.2),

    TimeDistributed(Dense(num_features))                 

])



model.compile(loss='mae', optimizer='adam')

model.summary()
es = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5)

history = model.fit(

    X_train, y_train,

    epochs=100,

    batch_size=32,

    validation_split=0.1,

    callbacks = [es],

    shuffle=False

)
plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend();
model.evaluate(X_test, y_test)
X_train_pred = model.predict(X_train)



train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])
train_mae_loss
sns.distplot(train_mae_loss, bins=50, kde=True);
X_test_pred = model.predict(X_test)



test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
sns.distplot(test_mae_loss, bins=50, kde=True);
THRESHOLD = 0.55



test_score_df = pd.DataFrame(test[time_steps:])

test_score_df['loss'] = test_mae_loss

test_score_df['threshold'] = THRESHOLD

test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

test_score_df['close'] = test[time_steps:].close
fig = go.Figure()

fig.add_trace(go.Scatter(x=test[time_steps:].date, y=test_score_df.loss,

                    mode='lines',

                    name='Test Loss'))

fig.add_trace(go.Scatter(x=test[time_steps:].date, y=test_score_df.threshold,

                    mode='lines',

                    name='Threshold'))

fig.update_layout(showlegend=True)

fig.show()
anomalies = test_score_df[test_score_df.anomaly == True]

anomalies.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=test[time_steps:].date, y=scaler.inverse_transform(test[time_steps:].close),

                    mode='lines',

                    name='Close Price'))

fig.add_trace(go.Scatter(x=anomalies.date, y=scaler.inverse_transform(anomalies.close),

                    mode='markers',

                    name='Anomaly'))

fig.update_layout(showlegend=True)

fig.show()