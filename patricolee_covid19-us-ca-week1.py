import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

train.Date = pd.to_datetime(train.Date)

train.info()
train
cc = train.ConfirmedCases.values

f = train.Fatalities.values

plt.plot(cc, color = 'blue', label = 'Confirmed Cases')

plt.plot(f, color = 'orange', label = 'Fatalities')

plt.legend()

plt.show()
plot_acf(cc)

plot_pacf(cc)

plt.show()
plot_acf(f)

plot_pacf(f)

plt.show()
ccf = train[['ConfirmedCases', 'Fatalities']]

ccf = ccf[ccf.ConfirmedCases > 0].values

ccf_max, ccf_min = np.max(ccf), np.min(ccf)

ccf_norm = (ccf - ccf_min) / (ccf_max - ccf_min)



X, y = [], []

for i in range(len(ccf_norm)):

    end = i+2

    if end > len(ccf_norm)-1:

        break

    X.append(ccf_norm[i:end])

    y.append(ccf_norm[end])

    

X, y = np.array(X).reshape(-1, 2, 2), np.array(y)

print(X.shape, y.shape)
tf.random.set_seed(1)



m = Sequential()

m.add(LSTM(100, input_shape = (X.shape[1], X.shape[2]), activation = 'relu'))

m.add(Dense(2))

m.compile(loss = 'mse', optimizer = 'adam')

h = m.fit(X, y, epochs = 100, verbose = 0)
plt.plot(h.history['loss'], label = 'Loss')

plt.legend()

plt.show()
plt.figure(figsize = (10, 4))

plt.subplot(121)

plt.title('Confirmed Cases')

plt.plot(np.pad((m.predict(X) * (ccf_max - ccf_min) + ccf_min)[:,0], (2,0)), 'r--', label = 'Predict')

plt.plot(ccf[:,0], label = 'Actual')

plt.legend()



plt.subplot(122)

plt.title('Fatalities')

plt.plot(np.pad((m.predict(X) * (ccf_max - ccf_min) + ccf_min)[:,1], (2,0)), 'r--', label = 'Predict')

plt.plot(ccf[:,1], label = 'Actual')

plt.legend()

plt.show()
test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

pd.concat([test.head(1), test.tail(1)])
test_pred = y[-2:]

n = len(test) - len(train[train.Date >= '2020-03-12'])

for i in range(n):

    p = m.predict(test_pred[-2:].reshape(-1, 2, 2))

    test_pred = np.append(test_pred, p).reshape(-1, 2)

    i += 1
test_pred_round = np.round(test_pred * (ccf_max - ccf_min) + ccf_min, 0)[:-2]

plt.plot(test_pred_round[:,0], label = 'Conf Cases Pred')

plt.plot(test_pred_round[:,1], label = 'Fatal Pred')

plt.legend()

plt.show()
sub = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
cc_sub = np.append(train[train.Date >= '2020-03-12'].ConfirmedCases.values, test_pred_round[:, 0])

f_sub = np.append(train[train.Date >= '2020-03-12'].Fatalities.values, test_pred_round[:, 1])
sub = sub.assign(ConfirmedCases = cc_sub, Fatalities = f_sub)

sub.to_csv('submission.csv', index = False)