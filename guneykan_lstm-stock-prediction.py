from pandas_datareader import data

import matplotlib.pyplot as plt

import pandas as pd

import pandas_datareader.data as web

import tensorflow as tf

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from functools import partial

import talib
def shuffle_batch(X, y, batch_size):

    rnd_idx = np.random.permutation(len(X))

    n_batches = len(X) // batch_size

    for batch_idx in np.array_split(rnd_idx, n_batches):

        X_batch, y_batch = X[batch_idx], y[batch_idx]

        yield X_batch, y_batch
start_date = '1970-12-31'

end_date = '2019-04-12'

df = web.DataReader('AAPL', 'yahoo', start_date, end_date)

df = df.drop(["Adj Close"], axis=1)

df["mid"] = (df["High"] + df["Low"]) / 2 # Predicting day avg prices are more robust
df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(np.array(df["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

df["cci"] = talib.CCI(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=20)

df["hilbert"] = talib.HT_DCPERIOD(np.array(df["Close"]))

df["hilbert1"] = talib.HT_DCPHASE(np.array(df["Close"]))

df["h_inphase"], df["h_quadrature"] = talib.HT_PHASOR(np.array(df["Close"]))

df["h_sine"], df["h_leadsine"] = talib.HT_SINE(np.array(df["Close"]))

df["h_integer"] = talib.HT_TRENDMODE(np.array(df["Close"]))



df["atr"] = talib.NATR(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14)



df["upperband"], df["middleband"], df["lowerband"] = talib.BBANDS(np.array(df["Close"]), timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)



df["mom120"] = talib.MOM(np.array(df["Close"]), timeperiod=120)

df["mom60"] = talib.MOM(np.array(df["Close"]), timeperiod=60)

df["mom20"] = talib.MOM(np.array(df["Close"]), timeperiod=20)

df["mom10"] = talib.MOM(np.array(df["Close"]), timeperiod=10)

df["ma20"] = talib.SMA(np.array(df["Close"]), timeperiod=20)

df["ma10"] = talib.SMA(np.array(df["Close"]), timeperiod=10)



df["roc"] = talib.ROC(np.array(df["Close"]), timeperiod=10)



df["ult"] = talib.ULTOSC(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod1=7, timeperiod2=14, timeperiod3=28)



df["will"] = talib.WILLR(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14)
df.describe()

df.info()

df.head()
df["mid_1df"] = df["mid"].shift(-1)
df = df.drop(["High", "Low", "Open", "Close", "mid"], axis=1)

df = df.dropna() # Also drop NaN values
train_df = df.iloc[:9000]

test_df = df.iloc[9000:9500]

train_df.tail()

test_df.head()
train_data_X = np.array(train_df.drop(["mid_1df"], axis=1).values)

train_data_y = np.array(train_df["mid_1df"].values)

test_data_X = np.array(test_df.drop(["mid_1df"], axis=1).values)

test_data_y = np.array(test_df["mid_1df"].values)
smoothing_window_size = 1000



scaler_min = MinMaxScaler()

for di in range(0,9000,smoothing_window_size):

    scaler_min.fit(train_data_X[di:di+smoothing_window_size,:])

    train_data_X[di:di+smoothing_window_size,:] = scaler_min.transform(train_data_X[di:di+smoothing_window_size,:])

    

del(scaler_min)

test_data_X = scaler_min.transform(test_data_X)
train_data_y = train_data_y.reshape(-1, 1)



scaler_1 = MinMaxScaler()



for di in range(0,9000,smoothing_window_size):

    scaler_1.fit(train_data_y[di:di+smoothing_window_size,:])

    train_data_y[di:di+smoothing_window_size,:] = scaler_1.transform(train_data_y[di:di+smoothing_window_size,:])



train_data_y = train_data_y.reshape(-1, )
X_train = train_data_X

y_train = train_data_y

X_test = test_data_X

y_test = test_data_y
n_steps = 50

n_inputs = X_train.shape[1]

n_neurons = 145

n_outputs = 1





# Reshape it for LSTM

X_test = X_test.reshape((-1, n_steps, n_inputs))

y_test = y_test.reshape((-1, n_steps, n_outputs))    



tf.reset_default_graph()



# Construction

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])



#keep_prob = tf.placeholder_with_default(1.0, shape=())



LSTM_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.elu, use_peepholes=True), 

                                                   output_size=n_outputs) 



#cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells]





outputs, states = tf.nn.dynamic_rnn(LSTM_cell, X, dtype=tf.float32)





learning_rate = 0.001







loss = tf.losses.absolute_difference(y, outputs)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)



init = tf.global_variables_initializer()

saver = tf.train.Saver()
max_checks_without_progress = 10

checks_without_progress = 0

best_loss = np.infty

n_iterations = 35

batch_size = 100

mse_train = [] # To check how loss function, changes with epochs, again I all add valid data set also but this version is so raw
with tf.Session() as sess:

    init.run()

    for iteration in range(n_iterations):

        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):

            X_batch = X_batch.reshape((-1, n_steps, n_inputs))

            y_batch = y_batch.reshape((-1, n_steps, n_outputs))

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})

        mse_train.append(mse)

        print(iteration, "\tMSE:", mse)

        if mse < best_loss: # adding stop loss

            save_path = saver.save(sess, "./my_LSTM_model_0_to_4.ckpt")

            best_loss = mse

            checks_without_progress = 0

        else:

            checks_without_progress += 1

            if checks_without_progress > max_checks_without_progress:

                print("Early stopping!")

                break

    X_new = X_test

    y_pred = sess.run(outputs, feed_dict={X: X_new}) # Prediction

    
plt.plot(mse_train)
y_pred = scaler_1.inverse_transform(y_pred.reshape(-1, 1))
plt.plot(y_test.reshape((-1, 1))[49::50])

plt.plot(y_pred.reshape((-1, 1))[49::50])



plt.plot(y_test.reshape((-1, 1)))

plt.plot(y_pred.reshape((-1, 1))) # One of the things that does not make sense to me when you check predictions every 50 steps its really accurate but squence itself not so much
comparison_change = pd.concat([pd.DataFrame(y_test.reshape((-1, 1))), 

                        pd.DataFrame(y_pred.reshape((-1, 1)))], axis=1)



# Calculate the pct change    

comparison_change = comparison_change.pct_change(1)

comparison_change = comparison_change.dropna()

comparison_change.columns = ["True", "Pred"]

comparison_change["True"].plot()

comparison_change["Pred"].plot()

# Tranform regression predictions to classification

test = []

for a in comparison_change["True"]:

    if a >= 0:

        test.append(1)

    else:

        test.append(0)



pred = []

for a in comparison_change["Pred"]:

    if a >= 0:

        pred.append(1)

    else:

        pred.append(0)



# Classification Report

print(classification_report(test, pred))