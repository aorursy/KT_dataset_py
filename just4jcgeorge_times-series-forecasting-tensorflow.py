import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline



width = 6

height = 3

import matplotlib

matplotlib.rcParams['figure.figsize'] = [width, height]



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import math

import pdb



#import tensorflow as tf - this uses tensorflow 2.0 which breaks the code

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



import sklearn

import sklearn.preprocessing



import numpy as np

import pandas as pd



from scipy.stats import pearsonr, spearmanr

## we download historical data from 1990-2019

## from Yahoo https://finance.yahoo.com/quote/%5EGSPC/history?period1=634885200&period2=1550034000&interval=1d&filter=history&frequency=1d
import os

df = pd.read_csv("/kaggle/input/SP500.csv")
df.shape
df.head()
df.tail()
## let's first take a look at our data

df.index = df.Date

fig = df.Close.plot()
(df.Close - df.Open).plot()
## we can see there have been several "regime changes"

## although it would be difficult to set an exact date of the change

## but do different weeks look all that different?
vals = df["1990-05-05":"1990-05-11"].Close.values

mean_val = np.mean(vals)

plt.plot([1, 2, 3, 4, 5], vals/mean_val)

plt.xticks([1, 2, 3, 4, 5])
vals = df["2000-05-05":"2000-05-11"].Close.values

mean_val = np.mean(vals)

plt.plot([1, 2, 3, 4, 5], vals/mean_val)

plt.xticks([1, 2, 3, 4, 5])
vals = df["2018-05-05":"2018-05-11"].Close.values

mean_val = np.mean(vals)

plt.plot([1, 2, 3, 4, 5], vals/mean_val)

plt.xticks([1, 2, 3, 4, 5])
## if we evaluate in terms of percent change within the week 

## none of these weeks seem distinctly different at the week-based scale to the eye
## We will use a deep learning approach, so we need to normalize our inputs to fall 

## within -1 to 1. we want to do so without letting information leak backwards from the future

## so we need to have a rolling smoothing process rather than taking the global mean to normalize

## these columns
## we want to predict daily returns (imagine you choose only to buy at start of day 

## and sell at end of day)
df.head()
df['Return'] = df.Close - df.Open
df.Return.plot()
df['DailyVolatility'] = df.High - df.Low
df.DailyVolatility.plot()
## as our inputs we will use daily volatility, daily return, and daily volume

## all should be scaled appropriately so we need to compute rolling means to scale these
## we will use an exponentially weighted moving average
ewdf = df.ewm(halflife = 10).mean()
ewdf.DailyVolatility.plot()
vewdf = df.ewm(halflife = 10).var()
## notice that we don't fit to the smoothed values we merely use them to 

((df.DailyVolatility - ewdf.DailyVolatility)/ vewdf.DailyVolatility**0.5 ).plot()
df['ScaledVolatility'] = ((df.DailyVolatility - ewdf.DailyVolatility)/ vewdf.DailyVolatility**0.5 )
df.head()
df['ScaledReturn'] = ((df.Return - ewdf.Return)/ vewdf.Return**0.5 )
df['ScaledVolume'] = ((df.Volume - ewdf.Volume)/ vewdf.Volume**0.5 )
df.head(12)
## remove first row, which has na

df = df.dropna()
## now we need to form input arrays and target arrays

## let's try to predict just a day ahead and see how we do

## predicting stock prices is notoriously difficult so we should not

## get ahead of ourselves
train_df = df[:7000]

test_df = df[7000:]

X = train_df[:(7000 - 10)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values

Y = train_df[10:]["ScaledReturn"].values

## however batches are usually in form TNC

## time, num examples, channels

## so we need to reshape
X.shape
X = np.expand_dims(X, axis = 1)
X.shape
X = np.split(X, X.shape[0]/10, axis = 0)
X = np.concatenate(X, axis = 1)
X.shape
X[:, 0, 1]
X[:, 1, 1]
X[:, 2, 1]
train_df[:(7000 - 10)][["ScaledReturn"]].values[:31]
Y_test = Y[::10]
Y_test[:3]
Y = Y_test
X.shape
Y.shape
## notice that we only used each data point once

## but actually each data point can belong to many series, occupying a different position in the series

## say it could be the first point or the last point or a middle point in the time series

## rather than explicitly expanding out, we will simply cut off a random number of points

## at each end so that for each epoch through training, we'll have different series
NUM_HIDDEN    = 8

NUM_LAYERS    = 1

LEARNING_RATE = 1e-2

EPOCHS        = 10

BATCH_SIZE    = 64

WINDOW_SIZE   = 20
Xinp = tf.placeholder(dtype = tf.float32, shape = [WINDOW_SIZE, None, 3])

Yinp = tf.placeholder(dtype = tf.float32, shape = [None])
with tf.variable_scope("scope1", reuse=tf.AUTO_REUSE):

    #rnn_cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, dtype = tf.float32)

    #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.9)

    #rnn_output, states = tf.nn.dynamic_rnn(rnn_cell, Xinp, dtype=tf.float32) 

    

    ## tf.nn.rnn_cell.MultiRNNCell

    cells = [tf.nn.rnn_cell.LSTMCell(num_units=NUM_HIDDEN) for n in range(NUM_LAYERS)]

    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    rnn_output, states = tf.nn.dynamic_rnn(stacked_rnn_cell, Xinp, dtype=tf.float32) 

    W = tf.get_variable("W_fc", [NUM_HIDDEN, 1], initializer = tf.random_uniform_initializer(-.2, .2))

    output = tf.squeeze(tf.matmul(rnn_output[-1, :, :], W))

    ## notice we have no bias because we expect average zero return

    loss = tf.nn.l2_loss(output - Yinp)

    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    ##opt = tf.train.AdamOptimizer(LEARNING_RATE)

    train_step = opt.minimize(loss)
## need to loop through data and find a way to jitter data 

## then need to also compute validation loss

## and need to record results
sess = tf.Session()

sess.run(tf.tables_initializer())

sess.run(tf.global_variables_initializer())
## for each epoch

y_hat_dict = {}

Y_dict = {}



in_sample_Y_dict = {}

in_sample_y_hat_dict = {}



for ep in range(EPOCHS):

    ## for each offset to create a new series of distinct time series 

    ## (re: overlapping issue we talked about previously)

    epoch_training_loss = 0.0

    for i in range(WINDOW_SIZE):

        X = train_df[:(7000 - WINDOW_SIZE)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values

        Y = train_df[WINDOW_SIZE:]["ScaledReturn"].values



        ## make it divisible by window size

        num_to_unpack = math.floor(X.shape[0] / WINDOW_SIZE)

        start_idx = X.shape[0] - num_to_unpack * WINDOW_SIZE

        X = X[start_idx:] 

        Y = Y[start_idx:]  

        

        X = X[i:-(WINDOW_SIZE-i)]

        Y = Y[i:-(WINDOW_SIZE-i)]                                

        

        X = np.expand_dims(X, axis = 1)

        X = np.split(X, X.shape[0]/WINDOW_SIZE, axis = 0)

        X = np.concatenate(X, axis = 1)

        Y = Y[::WINDOW_SIZE]

        ## TRAINING

        ## now batch it and run a sess

        for j in range(math.ceil(Y.shape[0] / BATCH_SIZE)):

            ll = BATCH_SIZE * j

            ul = BATCH_SIZE * (j + 1)

            

            if ul > X.shape[1]:

                ul = X.shape[1] - 1

                ll = X.shape[1]- BATCH_SIZE

            

            training_loss, _, y_hat = sess.run([loss, train_step, output],

                                       feed_dict = {

                                           Xinp: X[:, ll:ul, :], Yinp: Y[ll:ul]

                                       })

            epoch_training_loss += training_loss          

            

            in_sample_Y_dict[ep]     = Y[ll:ul] ## notice this will only net us the last part of data trained on

            in_sample_y_hat_dict[ep] = y_hat

            

        ## TESTING

        X = test_df[:(test_df.shape[0] - WINDOW_SIZE)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values

        Y = test_df[WINDOW_SIZE:]["ScaledReturn"].values

        num_to_unpack = math.floor(X.shape[0] / WINDOW_SIZE)

        start_idx = X.shape[0] - num_to_unpack * WINDOW_SIZE

        X = X[start_idx:] ## better to throw away beginning than end of training period when must delete

        Y = Y[start_idx:]                              

        

        X = np.expand_dims(X, axis = 1)

        X = np.split(X, X.shape[0]/WINDOW_SIZE, axis = 0)

        X = np.concatenate(X, axis = 1)

        Y = Y[::WINDOW_SIZE]

        testing_loss, y_hat = sess.run([loss, output],

                                 feed_dict = { Xinp: X, Yinp: Y })

        ## nb this is not great. we should really have a validation loss apart from testing

        

    print("Epoch: %d   Training loss: %0.2f   Testing loss %0.2f:" % (ep, epoch_training_loss, testing_loss))

    Y_dict[ep] = Y

    y_hat_dict[ep] = y_hat

    
plt.plot(Y_dict[EPOCHS - 1])

plt.plot(y_hat_dict[EPOCHS - 1], 'r')

plt.title('Out of sample performance')

plt.show()
plt.plot(in_sample_Y_dict[EPOCHS - 1])

plt.plot(in_sample_y_hat_dict[EPOCHS - 1], 'r')

plt.title('In sample performance')

plt.show()
pearsonr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1])
pearsonr(in_sample_Y_dict[EPOCHS - 1], in_sample_y_hat_dict[EPOCHS - 1])
spearmanr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1])
spearmanr(in_sample_Y_dict[EPOCHS - 1], in_sample_y_hat_dict[EPOCHS - 1])
plt.plot(Y_dict[EPOCHS - 1])

plt.plot(y_hat_dict[EPOCHS - 1] * 10, 'r')

plt.title('Rescaled out of sample performance')

plt.show()
plt.plot(in_sample_Y_dict[EPOCHS - 1])

plt.plot(in_sample_y_hat_dict[EPOCHS - 1] * 10, 'r')

plt.title('Rescaled in sample performance')

plt.show()
plt.plot(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1] * 10, linestyle="", marker="o")
pearsonr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1])
spearmanr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1])
in_sample_Y_dict[EPOCHS-1].shape