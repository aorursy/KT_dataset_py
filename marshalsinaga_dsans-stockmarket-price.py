# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.




apple = pd.read_csv('../input/apple_market_2.csv')

apple
for name in list(apple):

    print('statistic for column '+ name)

    print(apple[name].describe())

    print()
print(apple.isnull().sum())
import plotly.offline as py

import statsmodels.api as sm

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True)

%matplotlib inline



apple_data = apple

apple_data = apple_data.reset_index()

apple_data['date'] = pd.to_datetime(apple_data['Date'])

apple_data = apple_data.set_index('Date')



s = sm.tsa.seasonal_decompose(apple_data['WIKI/AAPL - Close'].values, freq=60)





trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',

    line = dict(color = ('rgb(244, 146, 65)'), width = 4))



trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',

    line = dict(color = ('rgb(66, 244, 155)'), width = 2))



trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',

    line = dict(color = ('rgb(209, 244, 66)'), width = 2))



trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',

    line = dict(color = ('rgb(66, 134, 244)'), width = 2))



data = [trace1, trace2, trace3, trace4]

layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='seasonal_decomposition')



import matplotlib.pyplot as plt

%matplotlib inline



for column in [column for column in list(apple) if column != 'Date']:

    apple[column].hist(bins=20)

    plt.xlabel(column)

    plt.ylabel('frequency')

    plt.show()
apple.skew(axis=0, skipna=True)
import seaborn as sns

for column in [column for column in list(apple) if column != 'Date']:

    apple_feature = apple[column]

    sns.boxplot(apple_feature)

    plt.xlabel(column)

    plt.show()
apple.corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
group = apple.groupby('Date')

Real_Price = group['WIKI/AAPL - Close'].mean()

Real_Price
N_STEPS_IN = 30

N_STEPS_OUT = 7

N_FEATURES = 1
apple_train= Real_Price[:-(N_STEPS_IN+N_STEPS_OUT)]

apple_test= Real_Price[-(N_STEPS_IN+N_STEPS_OUT):]

print(apple_train)

print(apple_test)
from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler()



train_set = apple_train.values

train_set = np.reshape(train_set, (len(train_set), 1))

train_set = sc.fit_transform(train_set)

train_set = np.reshape(train_set, (len(train_set),))



test_set = apple_test.values

test_set = np.reshape(test_set, (len(test_set), 1))

test_set = sc.transform(test_set)

test_set = np.reshape(test_set, (len(test_set),))
# split a univariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps_in

        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence

        if out_end_ix > len(sequence):

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)



X_train, y_train = split_sequence(train_set, N_STEPS_IN, N_STEPS_OUT)

X_test, y_test = split_sequence(test_set, N_STEPS_IN, N_STEPS_OUT)
X_train.shape
# reshaping X_train and X_test

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], N_FEATURES))

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], N_FEATURES))
# univariate multi-step vector-output stacked lstm example

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(N_STEPS_IN, N_FEATURES)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(N_STEPS_OUT))

model.compile(optimizer='adam', loss='mse')



fit = model.fit(X_train, y_train, epochs=50)



# predict

predicted_price = model.predict(X_test)

predicted_price = sc.inverse_transform(predicted_price)[0]

predicted_price
y_test_ = sc.inverse_transform(y_test)[0]

apple_test[-N_STEPS_OUT:].values
from matplotlib import pyplot as plt



plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()

ax.set_ylim([50,400])

plt.plot(y_test_, color = 'red', label = 'Real  Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Apple Price')

plt.title('Apple Price Prediction', fontsize=40)

apple_test_reset = apple_test[-N_STEPS_OUT:].reset_index()

x=apple_test_reset.index

labels = apple_test_reset['Date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Apple Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()





trace1 = go.Scatter(x=labels, y=y_test_, name= 'Actual Price',

                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))

trace2 = go.Scatter(x=labels, y=predicted_price, name= 'Predicted Price',

                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))

data = [trace1, trace2]

layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',

             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='results_demonstrating2')





trace1 = go.Scatter(

    x = np.arange(0, len(fit.history['loss']), 1),

    y = fit.history['loss'],

    mode = 'lines',

    name = 'Train loss',

    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')

)





data = [trace1]

layout = dict(title = 'Train loss during training',

              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='training_process')
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

print(mean_squared_error(y_test_,predicted_price))
score = model.evaluate(X_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[0], score*100))
from keras.layers import Bidirectional



model = Sequential()

model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(N_STEPS_IN, N_FEATURES)))

model.add(Dense(N_STEPS_OUT))

model.compile(optimizer='adam', loss='mse')



fit = model.fit(X_train, y_train, epochs=50)
# predict

predicted_price = model.predict(X_test)

predicted_price = sc.inverse_transform(predicted_price)[0]

predicted_price
y_test_ = sc.inverse_transform(y_test)[0]

apple_test[-N_STEPS_OUT:].values
from matplotlib import pyplot as plt



plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()

ax.set_ylim([50,400])

plt.plot(y_test_, color = 'red', label = 'Real  Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Apple Price')

plt.title('Apple Price Prediction', fontsize=40)

apple_test_reset = apple_test[-N_STEPS_OUT:].reset_index()

x=apple_test_reset.index

labels = apple_test_reset['Date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Apple Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()





trace1 = go.Scatter(x=labels, y=y_test_, name= 'Actual Price',

                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))

trace2 = go.Scatter(x=labels, y=predicted_price, name= 'Predicted Price',

                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))

data = [trace1, trace2]

layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',

             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='results_demonstrating2')
trace1 = go.Scatter(

    x = np.arange(0, len(fit.history['loss']), 1),

    y = fit.history['loss'],

    mode = 'lines',

    name = 'Train loss',

    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')

)





data = [trace1]

layout = dict(title = 'Train loss during training',

              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='training_process')
score = model.evaluate(X_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[0], score*100))

print(mean_squared_error(y_test_,predicted_price))

# define model 

model = Sequential()

model.add(LSTM(100, activation='relu'))

model.add(Dense(N_STEPS_OUT))

model.compile(optimizer='adam', loss='mse')



fit = model.fit(X_train, y_train, epochs=50)



# predict

predicted_price = model.predict(X_test)

predicted_price = sc.inverse_transform(predicted_price)[0]

predicted_price
from matplotlib import pyplot as plt



plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()

ax.set_ylim([50,400])

plt.plot(y_test_, color = 'red', label = 'Real  Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Apple Price')

plt.title('Apple Price Prediction', fontsize=40)

apple_test_reset = apple_test[-N_STEPS_OUT:].reset_index()

x=apple_test_reset.index

labels = apple_test_reset['Date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Apple Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()





trace1 = go.Scatter(x=labels, y=y_test_, name= 'Actual Price',

                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))

trace2 = go.Scatter(x=labels, y=predicted_price, name= 'Predicted Price',

                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))

data = [trace1, trace2]

layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',

             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='results_demonstrating2')
trace1 = go.Scatter(

    x = np.arange(0, len(fit.history['loss']), 1),

    y = fit.history['loss'],

    mode = 'lines',

    name = 'Train loss',

    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')

)





data = [trace1]

layout = dict(title = 'Train loss during training',

              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='training_process')
score = model.evaluate(X_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[0], score*100))

print(mean_squared_error(y_test_,predicted_price))

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D



n_features = 1

n_seq = 15

n_steps = 2



model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps,n_features)))

model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(100, activation='relu'))

model.add(Dense(N_STEPS_OUT))

model.compile(optimizer='adam', loss='mse')





X_train_2 = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))

fit = model.fit(X_train_2, y_train, epochs=50)
# predict

n_features = 1

n_seq = 15

n_steps = 2

X_test_2 = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))

predicted_price = model.predict(X_test_2)

predicted_price = sc.inverse_transform(predicted_price)[0]

predicted_price
from matplotlib import pyplot as plt



plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()

ax.set_ylim([50,400])

plt.plot(y_test_, color = 'red', label = 'Real  Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Apple Price')

plt.title('Apple Price Prediction', fontsize=40)

apple_test_reset = apple_test[-N_STEPS_OUT:].reset_index()

x=apple_test_reset.index

labels = apple_test_reset['Date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Apple Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()





trace1 = go.Scatter(x=labels, y=y_test_, name= 'Actual Price',

                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))

trace2 = go.Scatter(x=labels, y=predicted_price, name= 'Predicted Price',

                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))

data = [trace1, trace2]

layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',

             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='results_demonstrating2')
trace1 = go.Scatter(

    x = np.arange(0, len(fit.history['loss']), 1),

    y = fit.history['loss'],

    mode = 'lines',

    name = 'Train loss',

    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')

)





data = [trace1]

layout = dict(title = 'Train loss during training',

              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='training_process')
score = model.evaluate(X_test_2, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[0], score*100))

print(mean_squared_error(y_test_,predicted_price))

from keras.layers import ConvLSTM2D



# define model

model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))

model.add(Flatten())

model.add(Dense(N_STEPS_OUT))

model.compile(optimizer='adam', loss='mse')



X_train_2 = X_train.reshape((X_train.shape[0], n_seq, 1, n_steps, n_features))

fit = model.fit(X_train_2, y_train, epochs=50)
X_test_2 = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))

predicted_price = model.predict(X_test_2)

predicted_price = sc.inverse_transform(predicted_price)[0]

predicted_price
from matplotlib import pyplot as plt



plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()

ax.set_ylim([50,400])

plt.plot(y_test_, color = 'red', label = 'Real  Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Apple Price')

plt.title('Apple Price Prediction', fontsize=40)

apple_test_reset = apple_test[-N_STEPS_OUT:].reset_index()

x=apple_test_reset.index

labels = apple_test_reset['Date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(18)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Apple Price(USD)', fontsize=40)

plt.legend(loc=2, prop={'size': 25})

plt.show()





trace1 = go.Scatter(x=labels, y=y_test_, name= 'Actual Price',

                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))

trace2 = go.Scatter(x=labels, y=predicted_price, name= 'Predicted Price',

                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))

data = [trace1, trace2]

layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',

             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='results_demonstrating2')
trace1 = go.Scatter(

    x = np.arange(0, len(fit.history['loss']), 1),

    y = fit.history['loss'],

    mode = 'lines',

    name = 'Train loss',

    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')

)





data = [trace1]

layout = dict(title = 'Train loss during training',

              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename='training_process')
score = model.evaluate(X_test_2, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[0], score*100))

print(mean_squared_error(y_test_,predicted_price))
